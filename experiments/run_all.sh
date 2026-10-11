#!/bin/bash
# Run every experiment on a TrixiDocker pod (JULIA_PROJECT is preset there); one log per run.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
J="julia -t 8"
$J experiments/e1_basics.jl gpu                          > logs/e1_gpu.log 2>&1
$J experiments/e1_basics.jl cpu                          > logs/e1_cpu.log 2>&1
$J experiments/e2_dg1d_cuda.jl gpu 16384 Float32 100     > logs/e2_gpu_f32.log 2>&1
$J experiments/e2_dg1d_cuda.jl cpu 16384 Float32 100     > logs/e2_cpu_f32.log 2>&1
$J experiments/e2_dg1d_cuda.jl gpu 16384 Float64 100     > logs/e2_gpu_f64.log 2>&1
$J experiments/e2_dg1d_cuda.jl gpu 1048576 Float32 100   > logs/e2_gpu_f32_big.log 2>&1
$J experiments/e3_trixi_reactant.jl gpu raise 1          > logs/e3_gpu_raise.log 2>&1
$J experiments/e3_trixi_reactant.jl cpu raise 1          > logs/e3_cpu_raise.log 2>&1
