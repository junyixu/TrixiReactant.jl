# E3: upstream Trixi.jl (P4estMesh 2D, KernelAbstractions kernels) driven by Reactant.
# Steps are reported one by one, so the first thing that breaks is visible.
# usage (TrixiDocker image): julia experiments/e3_trixi_reactant.jl [gpu|cpu] [raise|noraise] [refinement_level]
using Trixi, Reactant, CUDA, KernelAbstractions, Printf

const BACKEND = get(ARGS, 1, "gpu")
const RAISE = get(ARGS, 2, "raise") == "raise"
const LEVEL = parse(Int, get(ARGS, 3, "1"))
const RealT = Float32
Reactant.set_default_backend(BACKEND)
println("backend=$BACKEND raise=$RAISE  Trixi threading=", Trixi._PREFERENCE_THREADING,
        "  loop_vectorization=", Trixi._PREFERENCE_LOOPVECTORIZATION)

function step(f, name)
    print(rpad(name, 46))
    try
        t = @elapsed r = f()
        @printf("ok   (%.1f s)\n", t)
        return r
    catch err
        println("FAIL")
        println("    ", first(split(sprint(showerror, err), '\n')))
        bt = catch_backtrace()
        for fr in first(stacktrace(bt), 12)
            s = string(fr)
            (occursin("Trixi", s) || occursin("Reactant", s) || occursin("KernelAbstractions", s)) &&
                println("      @ ", first(s, 150))
        end
        return nothing
    end
end

include(joinpath(@__DIR__, "reactant_overrides.jl"))

# --- elixir p4est_2d_dgsem/elixir_advection_basic.jl (setup part)
equations = LinearScalarAdvectionEquation2D((0.2, -0.7))
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
mesh = P4estMesh((8, 8), polydeg = 3, coordinates_min = (-1.0, -1.0),
                 coordinates_max = (1.0, 1.0), initial_refinement_level = LEVEL,
                 periodicity = true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver; boundary_conditions = boundary_condition_periodic)
tspan = (0.0, 1.0)

# --- CPU reference with the same KA kernels (Array storage)
ode_cpu = semidiscretize(semi, tspan; real_type = RealT)
u_cpu = copy(ode_cpu.u0)
du_cpu = similar(u_cpu)
Trixi.rhs_hyperbolic!(du_cpu, u_cpu, ode_cpu.p, zero(RealT))
println("nelements=", nelements(ode_cpu.p.solver, ode_cpu.p.cache), "  DOFs=", length(u_cpu))

# --- move to Reactant
semi_r = step("trixi_adapt(ConcreteRArray, Float32, semi)") do
    Trixi.trixi_adapt(Reactant.ConcreteRArray, RealT, ode_cpu.p)
end
u_r = Reactant.to_rarray(u_cpu)
du_r = Reactant.to_rarray(zero(u_cpu))

rhs_closure!(du, u) = (Trixi.rhs_hyperbolic!(du, u, semi_r, zero(RealT)); du)
rhs_semi!(du, u, semi) = (Trixi.rhs_hyperbolic!(du, u, semi, zero(RealT)); du)

hlo = semi_r === nothing ? nothing : step("@code_hlo rhs! (semi as argument)") do
    string(@code_hlo raise=RAISE rhs_semi!(du_r, u_r, semi_r))
end
if hlo !== nothing
    nops = count("stablehlo.", hlo)
    println("    stablehlo ops: $nops   kernel_call left: ",
            occursin("kernel_call", hlo) || occursin("jit_call", hlo),
            "   while loops: ", count("stablehlo.while", hlo))
end

f = semi_r === nothing ? nothing : step("@compile rhs! (semi as argument)") do
    @compile raise=RAISE sync=true rhs_semi!(du_r, u_r, semi_r)
end
if f === nothing && semi_r !== nothing
    f = step("@compile rhs! (semi captured in closure)") do
        @compile raise=RAISE sync=true rhs_closure!(du_r, u_r)
    end
    call = () -> f(du_r, u_r)
else
    call = () -> f(du_r, u_r, semi_r)
end

if f !== nothing
    step("run compiled rhs!") do
        call()
    end
    err = maximum(abs, Array(du_r) .- du_cpu) / maximum(abs, du_cpu)
    @printf("    relative error vs CPU KA rhs!: %.2e\n", err)
    tr = minimum(_ -> (t0 = time_ns(); call(); (time_ns() - t0) / 1e6), 1:20)
    tc = minimum(_ -> (t0 = time_ns(); Trixi.rhs_hyperbolic!(du_cpu, u_cpu, ode_cpu.p, zero(RealT)); (time_ns() - t0) / 1e6), 1:20)
    @printf("    rhs! time: Reactant %.3f ms   CPU (KA, %d threads) %.3f ms\n", tr, Threads.nthreads(), tc)
end
