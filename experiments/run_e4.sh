#!/bin/bash
# E4 on a TrixiDocker pod: every (mode, case) runs in its own process, so a crash only loses
# that case. Logs go to logs/e4/, progress is appended to logs/e4/progress.txt; each mode is
# mirrored to S3 when it finishes (tools/s3_upload.sh; RCLONE_UPLOAD=0 to skip).
# usage: [E4_MODES="gpu_noraise gpu_raise cpu_raise"] experiments/run_e4.sh [case...]
#        (default: every case in e4_kernel_inventory.jl, all three modes)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/e4
export RUN_NAME=${RUN_NAME:-$(date -u +%Y-%m-%dT%H%M)-$(hostname -s)}
cases=${*:-$(grep -oP '^\s*\("\K[0-9a-z_]+(?=", ex\()' experiments/e4_kernel_inventory.jl)}
modes=${E4_MODES:-gpu_noraise gpu_raise cpu_raise}
echo "# $(date -u +%FT%TZ) modes: $modes cases: $(echo $cases)" >> logs/e4/progress.txt
for mode in $modes; do
    backend=${mode%_*} raise=${mode#*_}
    for c in $cases; do
        log=logs/e4/${mode}_${c}.log skip=
        # a kernel that crashes the process (e.g. a segfault in a compiler pass) is named in
        # logs/e4/.current: keep that log as .crashN.txt and rerun the case without it
        for attempt in 1 2 3 4; do
            t0=$(date +%s)
            rm -f logs/e4/.current
            E4_SKIP=$skip E4_CURRENT=logs/e4/.current timeout 1800 \
                julia -t 8 experiments/e4_kernel_inventory.jl "$backend" "$raise" "$c" > "$log" 2>&1
            rc=$?
            echo "$backend $raise $c rc=$rc $(( $(date +%s) - t0 ))s${skip:+ skip=$skip}" >> logs/e4/progress.txt
            [ $rc -ne 0 ] && [ -s logs/e4/.current ] || break
            cp "$log" "logs/e4/${mode}_${c}.crash$attempt.txt"
            skip=$skip${skip:+,}$(cat logs/e4/.current)
        done
    done
    tools/s3_upload.sh logs/e4
done
echo "done $(date -u +%FT%TZ)" >> logs/e4/progress.txt
