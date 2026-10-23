#!/bin/bash
# Mirror experiment output with rclone, as CollisionOperators.jl/io.jl does. Since 2026-10-22 the
# target is Google Drive (the MPCDF S3 account was suspended); GDrive:mpcdf-s3 holds the old bucket's
# copy, so new runs go next to the old ones. (The name s3_upload.sh is kept for the callers.)
# Best-effort: a failed upload only warns, so it never aborts a run.
#
#   usage: tools/s3_upload.sh <file-or-dir>...
#     a file is copied to <remote>/<basename>, a directory's contents to <remote>/
#   RCLONE_UPLOAD=0  → do nothing (default on)
#   RCLONE_REMOTE=…  → destination (default GDrive:mpcdf-s3/trixi-reactant/<RUN_NAME>, '_' → '-')
#   RUN_NAME=…       → run directory in the bucket (default <UTC date>-<short hostname>)
#
# The `GDrive` remote comes from rclone.conf. On a pod or VM, copy just that section in without
# echoing it:
#   tools/px_put.sh <proxy-user> <(rclone config show GDrive) /root/.config/rclone/rclone.conf
set -u
[ "${RCLONE_UPLOAD:-1}" = 0 ] && exit 0
if ! command -v rclone > /dev/null; then
    echo "s3_upload: rclone not found, skipping" >&2
    exit 0
fi
run=${RUN_NAME:-$(date -u +%Y-%m-%d)-$(hostname -s)}
remote=${RCLONE_REMOTE:-GDrive:mpcdf-s3/trixi-reactant/${run//_/-}}
for path in "$@"; do
    if [ -d "$path" ]; then
        rclone copy --exclude ".*" "$path" "$remote"  # skip dotfiles (e.g. shell hook logs)
    else
        rclone copyto "$path" "$remote/$(basename "$path")"
    fi || echo "s3_upload: failed to upload $path to $remote" >&2
done
exit 0
