#!/bin/bash
# Mirror experiment output to S3 with rclone, as CollisionOperators.jl/io.jl does.
# Best-effort: a failed upload only warns, so it never aborts a run.
#
#   usage: tools/s3_upload.sh <file-or-dir>...
#     a file is copied to <remote>/<basename>, a directory's contents to <remote>/
#   RCLONE_UPLOAD=0  → do nothing (default on)
#   RCLONE_REMOTE=…  → destination (default mpcdf-s3:trixi-reactant/<RUN_NAME>, '_' → '-')
#   RUN_NAME=…       → run directory in the bucket (default <UTC date>-<short hostname>)
#
# The `mpcdf-s3` remote comes from rclone.conf. On a pod, copy just that section in without
# echoing it:
#   tools/px_put.sh <proxy-user> <(rclone config show mpcdf-s3) /root/.config/rclone/rclone.conf
# or pass RCLONE_CONFIG_MPCDF_S3_* environment variables to a container.
set -u
[ "${RCLONE_UPLOAD:-1}" = 0 ] && exit 0
if ! command -v rclone > /dev/null; then
    echo "s3_upload: rclone not found, skipping" >&2
    exit 0
fi
run=${RUN_NAME:-$(date -u +%Y-%m-%d)-$(hostname -s)}
remote=${RCLONE_REMOTE:-mpcdf-s3:trixi-reactant/${run//_/-}}
for path in "$@"; do
    if [ -d "$path" ]; then
        rclone copy --exclude ".*" "$path" "$remote"  # skip dotfiles (e.g. shell hook logs)
    else
        rclone copyto "$path" "$remote/$(basename "$path")"
    fi || echo "s3_upload: failed to upload $path to $remote" >&2
done
exit 0
