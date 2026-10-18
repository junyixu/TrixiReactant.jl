"""E23: the 3D flux-differencing volume integral of the E12 array prototype as one Pallas TPU kernel.

Same math as E12's `volume_fd_unrolled` for the compressible Euler equations with Trixi's
Kennedy-Gruber two-point flux on curved elements: for every node and every direction d,
    V[i] = sum_ii D_split[i, ii] * f_KG(u_i, u_ii, (Ja^d_i + Ja^d_ii) / 2)
along the node index of direction d. One kernel invocation handles a block of B elements; all
temporaries stay in VMEM, so HBM sees one read of u and Ja and one write of V (E12's XLA version
reads and writes 78-110 state-sized arrays per rhs!).

Arrays (row-major, elements last): u (5, n, n, n, E), ja (9, n, n, n, E) with ja[3 (d-1) + (j-1)] =
component j of Ja^d, D_split (n, n) (in SMEM), V (5, n, n, n, E).

usage (TPU host, /opt/jax-env/bin/python):
  e23_pallas_fd.py bench E [B]                    pure JAX on the TPU: both kernel variants vs the
                                                  same math in jnp (XLA)
  e23_pallas_fd.py lower E B out.mlir [variant]   StableHLO text (kernel as tpu_custom_call) for Reactant
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

N = 4          # nodes per direction (polydeg 3)
GAMMA = 1.4
SMEM = getattr(pltpu, "SMEM", None) or pltpu.MemorySpace.SMEM


def lgl_dsplit(n):
    """Trixi's `derivative_split` for n Legendre-Gauss-Lobatto nodes: 2 D, corners corrected."""
    if n == 4:
        x = np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0])
    else:
        raise ValueError("only n = 4")
    w = np.array([1 / np.prod([x[j] - x[k] for k in range(n) if k != j]) for j in range(n)])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / w[i] / (x[i] - x[j])
        D[i, i] = -D[i].sum()
    q = np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6])          # LGL weights, n = 4
    Ds = 2 * D
    Ds[0, 0] += 1 / q[0]
    Ds[-1, -1] -= 1 / q[-1]
    return Ds


def prim(u):
    rho = u[0]
    v1, v2, v3 = u[1] / rho, u[2] / rho, u[3] / rho
    p = (GAMMA - 1) * (u[4] - 0.5 * (u[1] * v1 + u[2] * v2 + u[3] * v3))
    return rho, v1, v2, v3, p, u[4] / rho


def kg(a, b, n1, n2, n3):
    """Kennedy-Gruber flux between primitive states a and b (rho, v1, v2, v3, p, e) along (n1, n2, n3)."""
    rho = 0.5 * (a[0] + b[0])
    v1, v2, v3 = 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]), 0.5 * (a[3] + b[3])
    p = 0.5 * (a[4] + b[4])
    e = 0.5 * (a[5] + b[5])
    vn = v1 * n1 + v2 * n2 + v3 * n3
    f1 = rho * vn
    return (f1, f1 * v1 + p * n1, f1 * v2 + p * n2, f1 * v3 + p * n3, f1 * e + p * vn)


def node_slice(x, d, i, lead=0):
    """slice i (kept, size 1) of node dimension d (0, 1, 2) of an array (lead..., n, n, n, B)."""
    idx = [slice(None)] * x.ndim
    idx[lead + d] = slice(i, i + 1)
    return x[tuple(idx)]


def kernel(ds_ref, u_ref, ja_ref, o_ref):
    u = [u_ref[v] for v in range(5)]                  # (n, n, n, B) each
    ja = [[ja_ref[3 * d + j] for j in range(3)] for d in range(3)]
    a = prim(u)
    acc = [jnp.zeros_like(u[0]) for _ in range(5)]
    for d in range(3):
        idx = jax.lax.broadcasted_iota(jnp.int32, u[0].shape, d)
        for ii in range(N):
            b = [node_slice(x, d, ii) for x in a]
            nrm = [0.5 * (ja[d][j] + node_slice(ja[d][j], d, ii)) for j in range(3)]
            f = kg(a, b, *nrm)
            coef = jnp.zeros_like(u[0])
            for i in range(N):
                coef = jnp.where(idx == i, ds_ref[i, ii], coef)
            acc = [acc[k] + coef * f[k] for k in range(5)]
    for v in range(5):
        o_ref[v] = acc[v]


def kernel_slab(ds_ref, u_ref, ja_ref, o_ref):
    """the same sums slab by slab: output node slab i of direction d, partner slab ii, scalar
    coefficients D_split[i, ii] (no iota/select), accumulated into the output ref"""
    u = [u_ref[v] for v in range(5)]
    ja = [[ja_ref[3 * d + j] for j in range(3)] for d in range(3)]
    a = prim(u)
    for v in range(5):
        o_ref[v] = jnp.zeros_like(u[0])
    for d in range(3):
        for i in range(N):
            ai = [node_slice(x, d, i) for x in a]
            jai = [node_slice(ja[d][j], d, i) for j in range(3)]
            acc = None
            for ii in range(N):
                if ii == i:                          # D_split has a zero diagonal
                    continue
                b = [node_slice(x, d, ii) for x in a]
                nrm = [0.5 * (jai[j] + node_slice(ja[d][j], d, ii)) for j in range(3)]
                term = [ds_ref[i, ii] * fk for fk in kg(ai, b, *nrm)]
                acc = term if acc is None else [x + y for x, y in zip(acc, term)]
            idx = [slice(None)] * 4
            idx[d] = slice(i, i + 1)
            for v in range(5):
                o_ref[(v,) + tuple(idx)] = o_ref[(v,) + tuple(idx)] + acc[v]


KERNELS = {"iota": kernel, "slab": kernel_slab}


def fd_pallas(ds, u, ja, B, variant="iota"):
    E = u.shape[-1]
    assert E % B == 0
    blk = lambda c: pl.BlockSpec((c, N, N, N, B), lambda e: (0, 0, 0, 0, e))
    return pl.pallas_call(
        KERNELS[variant],
        grid=(E // B,),
        in_specs=[pl.BlockSpec(memory_space=SMEM), blk(5), blk(9)],
        out_specs=blk(5),
        out_shape=jax.ShapeDtypeStruct(u.shape, u.dtype),
    )(ds, u, ja)


def fd_reference(ds, u, ja):
    """the same volume integral as plain jnp (XLA fuses it as it likes)"""
    a = prim([u[v] for v in range(5)])
    acc = [jnp.zeros_like(u[0]) for _ in range(5)]
    for d in range(3):
        for ii in range(N):
            b = [node_slice(x, d, ii) for x in a]
            nrm = [0.5 * (ja[3 * d + j] + node_slice(ja[3 * d + j], d, ii)) for j in range(3)]
            f = kg(a, b, *nrm)
            col = ds[:, ii].reshape([N if k == d else 1 for k in range(3)] + [1])
            acc = [acc[k] + col * f[k] for k in range(5)]
    return jnp.stack(acc)


def inputs(E, key=0):
    rng = np.random.default_rng(key)
    shape = (N, N, N, E)
    rho = 1 + 0.2 * rng.random(shape)
    v = 0.3 * (rng.random((3,) + shape) - 0.5)
    p = 1 + 0.2 * rng.random(shape)
    u = np.stack([rho, rho * v[0], rho * v[1], rho * v[2], p / (GAMMA - 1) + 0.5 * rho * (v ** 2).sum(0)])
    ja = 0.01 * (rng.random((9,) + shape) - 0.5)
    for d in range(3):
        ja[4 * d] += 1 / 64                           # (h/2)^2 for h = 1/4, diagonal of Ja
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    return f32(lgl_dsplit(N)), f32(u), f32(ja)


def timeit(f, *args, reps=20):
    f(*args).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*args).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def bench(E, B, variant):
    ds, u, ja = inputs(E)
    fp = jax.jit(lambda ds, u, ja: fd_pallas(ds, u, ja, B, variant))
    fr = jax.jit(fd_reference)
    try:
        t0 = time.perf_counter()
        vp = fp(ds, u, ja).block_until_ready()
        tcp = time.perf_counter() - t0
    except Exception as e:
        print(f"E = {E}, B = {B}, {variant}: FAIL {str(e).splitlines()[0][:400]}")
        return
    vr = fr(ds, u, ja).block_until_ready()
    err = float(jnp.max(jnp.abs(vp - vr)) / jnp.max(jnp.abs(vr)))
    tp, tr = timeit(fp, ds, u, ja), timeit(fr, ds, u, ja)
    bytes_min = 4 * (5 + 9 + 5) * N ** 3 * E          # read u and Ja, write V once
    print(f"E = {E} elements, B = {B}, kernel {variant}: max|V_pallas - V_xla| / max|V| = {err:.1e}")
    print(f"  Pallas kernel {tp:8.3f} ms (compile {tcp:.1f} s), {bytes_min / tp / 1e6:.0f} GB/s for the minimal traffic")
    print(f"  jnp / XLA     {tr:8.3f} ms")
    print(f"CSV,e23_bench,{E},{B},{variant},{err:.3e},{tp:.4f},{tr:.4f}", flush=True)


def lower(E, B, path, variant):
    spec = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    f = jax.jit(lambda ds, u, ja: fd_pallas(ds, u, ja, B, variant))
    text = f.trace(spec(N, N), spec(5, N, N, N, E), spec(9, N, N, N, E)).lower(lowering_platforms=("tpu",)).as_text()
    assert "tpu_custom_call" in text
    with open(path, "w") as io:
        io.write(text)
    print(f"wrote {path} ({len(text)} characters) for E = {E}, B = {B}")


if __name__ == "__main__":
    task = sys.argv[1]
    print("jax", jax.__version__, "devices", jax.devices())
    if task == "bench":
        E = int(sys.argv[2])
        for B in ([int(sys.argv[3])] if len(sys.argv) > 3 else [128, 256, 512]):
            for variant in KERNELS:
                bench(E, B, variant)
    elif task == "lower":
        lower(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "iota")
    else:
        raise SystemExit(__doc__)
