"""E26: the whole rhs! of the E12 brick path as one Pallas TPU kernel.

E23 moved the flux-differencing volume integral into a Pallas kernel (2.19 ms instead of 27.6 ms),
but the rest of the rhs! (faces, surface integral, source terms, layout conversions) still took
about 30 ms in XLA. Here one kernel does everything for compressible Euler 3D on a brick mesh with
Trixi's `elixir_euler_source_terms_nonperiodic.jl` setup:

    du = -J^-1 (V + factor S) + s
    V  flux-differencing volume integral, Kennedy-Gruber two-point flux (as E23)
    S  surface integral: on every element face the local Lax-Friedrichs flux
       (`FluxLaxFriedrichs(max_abs_speed_naive)`) between the own face values and the neighbour's
       (Dirichlet values `initial_condition_convergence_test` on the domain boundary), with the own
       outward normal +-Ja^d at the face nodes
    s  `source_terms_convergence_test`

Arrays are the brick path's: one (n, n, n, Nx, Ny, Nz) array per component (nodes i, j, k, then the
element grid), row-major, so the TPU tiles the (Ny, Nz) plane. One grid step handles one x-slab of
Ny x Nz elements: y and z neighbours are in the same block (shifted with `pltpu.roll`), the x
neighbours' face layers come in as two extra one-layer blocks of u. HBM sees u (plus two face
layers per slab), Ja, J^-1 and the coordinates once, and du once.

The `stage` variant does one Carpenter-Kennedy RK stage in the same kernel: with scalars
(t, dt, a, b) it returns K' = a K + dt du(U, t) and U' = U + b K', so du never goes to HBM.

usage (TPU host, /opt/jax-env/bin/python):
  e26_pallas_fused.py bench Nx Ny Nz                    pure JAX on the TPU: kernel vs the same math in jnp
  e26_pallas_fused.py lower Nx Ny Nz out.mlir [stage]   StableHLO text (kernel as tpu_custom_call) for Reactant
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e23_pallas_fd import N, GAMMA, SMEM, lgl_dsplit, prim, kg, node_slice, timeit

FACTOR = 6.0                     # Trixi's `inverse_weights[1]` for n = 4 (LGL weight 1/6)
OMEGA = float(np.pi)             # convergence test: 2 pi f with f = 1 / L, L = 2
NIN = 5 + 9 + 1 + 3              # u, Ja, J^-1, x


def layer(x, d, i):
    """node layer i of node dimension d of an array (n, n, n, ...), that dimension dropped"""
    return x[(slice(None),) * d + (i,)]


def flux_normal(u, a, n1, n2, n3):
    """Trixi's `flux(u, normal_direction, ::CompressibleEulerEquations3D)`; a = prim(u)"""
    rho, v1, v2, v3, p = a[0], a[1], a[2], a[3], a[4]
    vn = v1 * n1 + v2 * n2 + v3 * n3
    rvn = rho * vn
    return (rvn, rvn * v1 + p * n1, rvn * v2 + p * n2, rvn * v3 + p * n3, (u[4] + p) * vn)


def llf(ul, ur, n1, n2, n3):
    """`FluxLaxFriedrichs(max_abs_speed_naive)`: central flux minus 0.5 lambda (u_rr - u_ll)"""
    al, ar = prim(ul), prim(ur)
    fl, fr = flux_normal(ul, al, n1, n2, n3), flux_normal(ur, ar, n1, n2, n3)
    vl = al[1] * n1 + al[2] * n2 + al[3] * n3
    vr = ar[1] * n1 + ar[2] * n2 + ar[3] * n3
    cl = jnp.sqrt(GAMMA * al[4] / al[0])
    cr = jnp.sqrt(GAMMA * ar[4] / ar[0])
    lam = jnp.maximum(jnp.abs(vl), jnp.abs(vr)) + jnp.maximum(cl, cr) * jnp.sqrt(n1 * n1 + n2 * n2 + n3 * n3)
    return [0.5 * (fl[k] + fr[k]) - 0.5 * lam * (ur[k] - ul[k]) for k in range(5)]


def boundary_values(x1, x2, x3, t):
    """`initial_condition_convergence_test(x, t, ::CompressibleEulerEquations3D)`"""
    ini = 2.0 + 0.1 * jnp.sin(OMEGA * (x1 + x2 + x3 - t))
    return [ini, ini, ini, ini, ini * ini]


def source_terms(x1, x2, x3, t):
    """`source_terms_convergence_test(u, x, t, ::CompressibleEulerEquations3D)`"""
    arg = OMEGA * (x1 + x2 + x3 - t)
    si, co = jnp.sin(arg), jnp.cos(arg)
    rho = 2.0 + 0.1 * si
    rho_x = OMEGA * 0.1 * co
    tmp = (2.0 * rho - 1.5) * (GAMMA - 1)
    du2 = rho_x * (2.0 + tmp)
    return [2.0 * rho_x, du2, du2, du2, rho_x * (4.0 * rho + 3.0 * tmp)]


def volume_into(o_r, ds_ref, a, ja):
    """E23's slab sums: o_r[v] = flux-differencing volume integral (zero diagonal of D_split skipped)"""
    for v in range(5):
        o_r[v][...] = jnp.zeros_like(a[0])
    for d in range(3):
        for i in range(N):
            ai = [node_slice(x, d, i) for x in a]
            jai = [node_slice(ja[d][j], d, i) for j in range(3)]
            acc = None
            for ii in range(N):
                if ii == i:
                    continue
                b = [node_slice(x, d, ii) for x in a]
                nrm = [0.5 * (jai[j] + node_slice(ja[d][j], d, ii)) for j in range(3)]
                term = [ds_ref[i, ii] * fk for fk in kg(ai, b, *nrm)]
                acc = term if acc is None else [x + y for x, y in zip(acc, term)]
            idx = [slice(None)] * 5
            idx[d] = slice(i, i + 1)
            for v in range(5):
                o_r[v][tuple(idx)] = o_r[v][tuple(idx)] + acc[v]


def make_kernel(Nx, Ny, Nz, stage=False):
    def kernel(ds_ref, t_ref, *refs):
        u_r, hm_r, hp_r = refs[0:5], refs[5:10], refs[10:15]
        if stage:                                                  # + K in, U' and K' out (K' accumulates)
            k_r, refs = refs[15:20], refs[0:15] + refs[20:]
        ja_r, ij_r, x_r = refs[15:24], refs[24], refs[25:28]
        o_r = refs[33:38] if stage else refs[28:33]
        t = t_ref[0]
        xb = pl.program_id(0)
        u = [r[...] for r in u_r]                                  # (n, n, n, Ny, Nz)
        ja = [[ja_r[3 * d + j][...] for j in range(3)] for d in range(3)]
        X = [r[...] for r in x_r]
        volume_into(o_r, ds_ref, prim(u), ja)
        # surface integral, face by face
        for d in range(3):
            for side in (0, 1):
                li = 0 if side == 0 else N - 1
                own = [layer(q, d, li) for q in u]                 # (n, n, Ny, Nz)
                sgn = -1.0 if side == 0 else 1.0
                nrm = [sgn * layer(ja[d][j], d, li) for j in range(3)]
                xs = [layer(q, d, li) for q in X]
                bv = boundary_values(*xs, t)
                if d == 0:                                         # x: neighbour layer from the next slab
                    nbr = [(hm_r if side == 0 else hp_r)[c][...] for c in range(5)]
                    at = (xb == 0) if side == 0 else (xb == Nx - 1)
                    w = jnp.where(at, 1.0, 0.0).astype(jnp.float32)
                    nbr = [w * bv[c] + (1.0 - w) * nbr[c] for c in range(5)]
                else:                                              # y, z: neighbour in this block
                    axis, size = (2, Ny) if d == 1 else (3, Nz)
                    other = [layer(q, d, N - 1 - li) for q in u]
                    shift = 1 if side == 0 else size - 1          # -face: from element - 1
                    nbr = [pltpu.roll(q, shift, axis) for q in other]
                    pos = jax.lax.broadcasted_iota(jnp.int32, own[0].shape, axis)
                    edge = pos == (0 if side == 0 else size - 1)
                    nbr = [jnp.where(edge, bv[c], nbr[c]) for c in range(5)]
                f = llf(own, nbr, *nrm)
                idx = (slice(None),) * d + (li,)
                for c in range(5):
                    o_r[c][idx] = o_r[c][idx] + FACTOR * f[c]
        s = source_terms(*X, t)
        ij = ij_r[...]
        if stage:
            dt, a, b = t_ref[1], t_ref[2], t_ref[3]
            uo_r = refs[28:33]
            for c in range(5):
                kn = a * k_r[c][...] + dt * (-ij * o_r[c][...] + s[c])
                o_r[c][...] = kn
                uo_r[c][...] = u[c] + b * kn
        else:
            for c in range(5):
                o_r[c][...] = -ij * o_r[c][...] + s[c]
    return kernel


def vmem_params(limit):
    for name in ("CompilerParams", "TPUCompilerParams"):
        cls = getattr(pltpu, name, None)
        if cls is not None:
            return cls(vmem_limit_bytes=limit)
    return None


def rhs_fused(ds, t, *arrs, vmem=96 * 2**20, stage=False):
    """arrs: u (5), Ja (9, Ja[3(d-1) + j] = component j of Ja^d), J^-1, x (3); each (n, n, n, Nx, Ny, Nz).
    stage: t = (t, dt, a, b) and arrs = u (5), K (5), Ja, J^-1, x; returns U' (5), K' (5)"""
    u = arrs[0:5]
    rest = arrs[10:] if stage else arrs[5:]
    n, _, _, Nx, Ny, Nz = u[0].shape
    full = pl.BlockSpec((n, n, n, None, Ny, Nz), lambda x: (0, 0, 0, x, 0, 0))
    lo = pl.BlockSpec((None, n, n, None, Ny, Nz), lambda x: (n - 1, 0, 0, jnp.maximum(x - 1, 0), 0, 0))
    hi = pl.BlockSpec((None, n, n, None, Ny, Nz), lambda x: (0, 0, 0, jnp.minimum(x + 1, Nx - 1), 0, 0))
    smem = pl.BlockSpec(memory_space=SMEM)
    kw = {}
    cp = vmem_params(vmem)
    if cp is not None:
        kw["compiler_params"] = cp
    nk, nout = (5, 10) if stage else (0, 5)
    out = pl.pallas_call(
        make_kernel(Nx, Ny, Nz, stage),
        grid=(Nx,),
        in_specs=[smem, smem] + [full] * 5 + [lo] * 5 + [hi] * 5 + [full] * (nk + 9 + 1 + 3),
        out_specs=[full] * nout,
        out_shape=[jax.ShapeDtypeStruct(u[0].shape, u[0].dtype)] * nout,
        **kw,
    )(ds, t, *u, *u, *u, *arrs[5:5 + nk], *rest)
    return tuple(out)


def stage_reference(ds, sc, *arrs):
    u, k = arrs[0:5], arrs[5:10]
    du = rhs_reference(ds, sc[0:1], *u, *arrs[10:])
    kn = [sc[2] * k[c] + sc[1] * du[c] for c in range(5)]
    return tuple(u[c] + sc[3] * kn[c] for c in range(5)) + tuple(kn)


def rhs_reference(ds, t, *arrs):
    """the same rhs! as plain jnp (XLA)"""
    u, ja, ij, X = arrs[0:5], arrs[5:14], arrs[14], arrs[15:18]
    a = prim(u)
    acc = [jnp.zeros_like(u[0]) for _ in range(5)]
    for d in range(3):
        for ii in range(N):
            b = [node_slice(x, d, ii) for x in a]
            nrm = [0.5 * (ja[3 * d + j] + node_slice(ja[3 * d + j], d, ii)) for j in range(3)]
            f = kg(a, b, *nrm)
            col = ds[:, ii].reshape([N if k == d else 1 for k in range(3)] + [1, 1, 1])
            acc = [acc[k] + col * f[k] for k in range(5)]
    for d in range(3):
        axis = 2 + d                                    # element axis of direction d in a layer array
        for side in (0, 1):
            li = 0 if side == 0 else N - 1
            own = [layer(q, d, li) for q in u]
            other = [layer(q, d, N - 1 - li) for q in u]
            nbr = [jnp.roll(q, 1 if side == 0 else -1, axis) for q in other]
            size = own[0].shape[axis]
            pos = jax.lax.broadcasted_iota(jnp.int32, own[0].shape, axis)
            edge = pos == (0 if side == 0 else size - 1)
            bv = boundary_values(*[layer(q, d, li) for q in X], t[0])
            nbr = [jnp.where(edge, bv[c], nbr[c]) for c in range(5)]
            sgn = -1.0 if side == 0 else 1.0
            f = llf(own, nbr, *[sgn * layer(ja[3 * d + j], d, li) for j in range(3)])
            idx = (slice(None),) * d + (li,)
            acc = [acc[c].at[idx].add(FACTOR * f[c]) for c in range(5)]
    s = source_terms(*X, t[0])
    return tuple(-ij * acc[c] + s[c] for c in range(5))


def inputs(Nx, Ny, Nz, t=0.3, key=0):
    """a uniform Cartesian brick mesh of [0, 2]^3 (nodes at LGL points) and a smooth physical state"""
    rng = np.random.default_rng(key)
    xi = np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0])
    h = 2.0 / np.array([Nx, Ny, Nz])
    shape = (N, N, N, Nx, Ny, Nz)

    def coord(a, Ne):
        c = (np.arange(Ne) + 0.5) * h[a]
        x = c[None, :] + 0.5 * h[a] * xi[:, None]          # (n, Ne)
        sh = [1] * 6
        sh[a], sh[3 + a] = N, Ne
        return np.broadcast_to(x.reshape(sh), shape)

    X = [coord(0, Nx), coord(1, Ny), coord(2, Nz)]
    ini = 2.0 + 0.1 * np.sin(OMEGA * (X[0] + X[1] + X[2] - t))
    u = [ini * (1 + 0.01 * rng.standard_normal(shape)) for _ in range(4)] + [ini ** 2]
    ja = []
    for d in range(3):
        for j in range(3):
            base = (h[(d + 1) % 3] * h[(d + 2) % 3] / 4) if j == d else 0.0
            ja.append(base + 1e-4 * rng.standard_normal(shape))
    ij = np.full(shape, 8 / (h[0] * h[1] * h[2]))
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    return f32(lgl_dsplit(N)), f32([t]), [f32(q) for q in u + ja + [ij] + X]


def bench(Nx, Ny, Nz):
    ds, t, arrs = inputs(Nx, Ny, Nz)
    E = Nx * Ny * Nz
    fp = jax.jit(rhs_fused)
    fr = jax.jit(rhs_reference)
    t0 = time.perf_counter()
    try:
        vp = jax.block_until_ready(fp(ds, t, *arrs))
    except Exception as e:
        print(f"{Nx}x{Ny}x{Nz}: FAIL {str(e)[:2000]}")
        return
    tcp = time.perf_counter() - t0
    vr = jax.block_until_ready(fr(ds, t, *arrs))
    err = max(float(jnp.max(jnp.abs(p - r)) / jnp.max(jnp.abs(r))) for p, r in zip(vp, vr))
    stack = lambda f: (lambda *a: jnp.stack(f(*a)))
    tp, tr = timeit(jax.jit(stack(rhs_fused)), ds, t, *arrs), timeit(jax.jit(stack(rhs_reference)), ds, t, *arrs)
    bytes_min = 4 * E * (N ** 3 * (NIN + 5) + N ** 2 * 2 * 5)      # + the two x face layers of u
    print(f"{Nx}x{Ny}x{Nz} = {E} elements: max|du_pallas - du_xla| / max|du| = {err:.1e}")
    print(f"  Pallas fused rhs! {tp:8.3f} ms (compile {tcp:.1f} s), {bytes_min / tp / 1e6:.0f} GB/s for the minimal traffic")
    print(f"  jnp / XLA         {tr:8.3f} ms")
    print(f"CSV,e26_bench,{Nx},{Ny},{Nz},{err:.3e},{tp:.4f},{tr:.4f}", flush=True)
    # one RK stage in the kernel
    sc = jnp.asarray([0.3, 1e-3, -0.5, 0.25], dtype=jnp.float32)
    ks = [0.1 * q for q in arrs[0:5]]
    sargs = arrs[0:5] + ks + arrs[5:]
    fs = jax.jit(lambda ds, sc, *a: rhs_fused(ds, sc, *a, stage=True))
    vs = jax.block_until_ready(fs(ds, sc, *sargs))
    vrs = jax.block_until_ready(jax.jit(stage_reference)(ds, sc, *sargs))
    errs = max(float(jnp.max(jnp.abs(p - r)) / jnp.max(jnp.abs(r))) for p, r in zip(vs, vrs))
    ts = timeit(jax.jit(lambda *a: jnp.stack(fs(*a))), ds, sc, *sargs)
    print(f"  Pallas RK stage   {ts:8.3f} ms, max rel. diff vs jnp {errs:.1e}")


def lower(Nx, Ny, Nz, path, stage=False):
    spec = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    a = spec(N, N, N, Nx, Ny, Nz)
    f = jax.jit(lambda ds, t, *x: rhs_fused(ds, t, *x, stage=stage))
    sc, nin = (spec(4), NIN + 5) if stage else (spec(1), NIN)
    text = f.trace(spec(N, N), sc, *([a] * nin)).lower(lowering_platforms=("tpu",)).as_text()
    assert "tpu_custom_call" in text
    with open(path, "w") as io:
        io.write(text)
    print(f"wrote {path} ({len(text)} characters) for {Nx}x{Ny}x{Nz} elements")


if __name__ == "__main__":
    task = sys.argv[1]
    print("jax", jax.__version__, "devices", jax.devices())
    dims = tuple(int(s) for s in sys.argv[2:5])
    if task == "bench":
        bench(*dims)
    elif task == "lower":
        lower(*dims, sys.argv[5], len(sys.argv) > 6 and sys.argv[6] == "stage")
    else:
        raise SystemExit(__doc__)
