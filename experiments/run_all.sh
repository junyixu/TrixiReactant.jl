#!/bin/bash
# Run every experiment on a TrixiDocker pod (JULIA_PROJECT is preset there); one log per run.
# Each log is mirrored to S3 as soon as its run ends (tools/s3_upload.sh; RCLONE_UPLOAD=0 to skip).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export RUN_NAME=${RUN_NAME:-$(date -u +%Y-%m-%dT%H%M)-$(hostname -s)}
J="julia -t 8"
run() {
    local log=logs/$1.log
    shift
    $J "$@" > "$log" 2>&1
    tools/s3_upload.sh "$log"
}
run e1_gpu          experiments/e1_basics.jl gpu
run e1_cpu          experiments/e1_basics.jl cpu
run e2_gpu_f32      experiments/e2_dg1d_cuda.jl gpu 16384 Float32 100
run e2_cpu_f32      experiments/e2_dg1d_cuda.jl cpu 16384 Float32 100
run e2_gpu_f64      experiments/e2_dg1d_cuda.jl gpu 16384 Float64 100
run e2_gpu_f32_big  experiments/e2_dg1d_cuda.jl gpu 1048576 Float32 100
run e3_gpu_raise    experiments/e3_trixi_reactant.jl gpu raise 1
run e3_cpu_raise    experiments/e3_trixi_reactant.jl cpu raise 1
experiments/run_e4.sh  # one process per (mode, case), logs in logs/e4/
