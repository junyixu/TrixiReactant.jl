# E19 smoke test: does a Pallas TPU kernel written in JAX run inside a Reactant program on the TPU?
# jax (CPU platform, so it does not open the TPU) lowers the kernel for TPU; the StableHLO, with the
# kernel as a `tpu_custom_call`, goes into a Reactant function via `Ops.hlo_call`, and Reactant
# compiles and runs it with its own libtpu. `jax` as a second argument instead runs the same
# kernel with JAX itself on the TPU (its own libtpu; a separate process, JAX_PLATFORMS=tpu).
# usage (TrixiDocker image :jax-tpu, on a TPU VM): julia experiments/e19_pallas_smoke.jl [reactant|jax]
const MODE = get(ARGS, 1, "reactant")
ENV["JAX_PLATFORMS"] = MODE == "jax" ? "tpu" : "cpu"
using PythonCall

pyexec("""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel(x_ref, o_ref):
    o_ref[...] = 2 * x_ref[...] + 1

def f(x):
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

spec = jax.ShapeDtypeStruct((256, 128), jnp.float32)
text = jax.jit(f).trace(spec).lower(lowering_platforms=("tpu",)).as_text()
""", Main)
println("jax ", pyeval(String, "jax.__version__", Main), ", devices ", pyeval(String, "str(jax.devices())", Main))

if MODE == "jax"
    err = pyeval(Float64, "float(abs(f(x := jnp.arange(256 * 128, dtype=jnp.float32).reshape(256, 128) / 1000) - (2 * x + 1)).max())", Main)
    println("JAX on the TPU: max error ", err)
    exit(0)
end

# separate top-level statements from here: `@jit` only expands once Reactant is loaded
using Reactant
Reactant.set_default_backend("tpu")
const TEXT = pyeval(String, "text", Main)
occursin("tpu_custom_call", TEXT) || error("no tpu_custom_call in the lowered module")
g(x) = Reactant.Ops.hlo_call(TEXT, x)[1]
# Reactant keeps Julia's dimension order: a (256, 128) array is the tensor<256x128> of the kernel
x = reshape(Float32.(0:(256 * 128 - 1)) ./ 1000, 256, 128)
y = @jit g(Reactant.to_rarray(x))
println("Reactant on the TPU: max error ", maximum(abs, Array(y) .- (2 .* x .+ 1)))
