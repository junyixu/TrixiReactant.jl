#!/usr/bin/env python3
"""Summarize E4 logs (logs/e4/<backend>_<raise>_<case>.log) as Markdown tables.

usage: tools/e4_summary.py [logdir]     (default: logs/e4)

Runs where the logs are (e.g. on the pod), so only the small summary has to be copied.
Statuses come from the table each run prints at the end; failure reasons from the section
of the log below each kernel (stderr can interleave with the kernel line, so the reason is
searched in the whole section).
"""
import re
import sys
from pathlib import Path

MODES = ["gpu_noraise", "gpu_raise", "cpu_raise"]
SHORT = {"ok (raised)": "raised", "ok (kernel kept)": "kept", "MISMATCH": "MISMATCH"}

logdir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/e4")
row_re = re.compile(r"^(\S+)\s+(\S+)\s+(.+?)\s+(\S+)\s+(\S+)$")
kernel_start = re.compile(r"^  ([A-Za-z0-9_]+!)")


def crash_reason(text):
    """Name the signal and the MLIR pass/pattern it happened in, if any."""
    sig = re.search(r"signal (\d+) \(\d+\): (.+)", text)
    if not sig:
        return "crash/timeout without a signal"
    after = text[sig.end():]
    pattern = re.search(r"([A-Z][A-Za-z]+)\d+matchAndRewrite", after)
    pass_ = re.search(r"([A-Z][A-Za-z]+Pass)\d+runOnOperation", after)
    where = " in " + "/".join(x[1] for x in (pass_, pattern) if x) if (pass_ or pattern) else ""
    return f"{sig[2].strip()}{where}"


def raise_reason(section, lines, kernel):
    """Which op could not be raised (the op may be printed on the line after the error)."""
    for i, l in enumerate(section):
        if "cannot raise op" in l:
            near = " ".join(section[i:i + 2])
            op = re.search(r'"([a-z_]+\.[a-z_.]+)"', near)
            callee = re.search(r"callee = @(\w+)", near)
            elem = re.search(r"elem_type = (i\d+|f\d+|!llvm\.struct)", near)
            what = f"`{op[1]}`" if op else "an op"
            what += f" @{callee[1]}" if callee else ""
            what += f" of {elem[1].replace('!llvm.', '')}" if elem else ""
            return f"cannot raise {what} to StableHLO"
    base = kernel.rstrip("!")
    if any("failed to raise func" in l and f"gpu_{base}" in l for l in lines):
        return "failed to raise func (no single op reported)"
    return None


def clean(reason):
    reason = re.sub(r"%\d+ = ", "", reason)
    reason = re.sub(r"<\{.*", "", reason)
    reason = re.sub(r"\(%.*", "", reason)
    reason = re.sub(r"\s+", " ", reason).strip(" |")
    return reason[:110]


def sections(lines):
    """kernel name -> lines of its section in the log body"""
    out, cur = {}, None
    for line in lines:
        if line.startswith("backend="):
            break
        m = kernel_start.match(line)
        if m:
            cur = m[1]
            out.setdefault(cur, [])
        if cur:
            out[cur].append(line)
    return out


results = {}  # (case, kernel) -> {mode: cell}
reasons = {}  # failure reason -> set of (mode, kernel)
cases = []
for log in sorted(logdir.glob("*.log")):
    m = re.match(r"(gpu|cpu)_(raise|noraise)_(.+)\.log$", log.name)
    if not m:
        continue
    mode, case = f"{m[1]}_{m[2]}", m[3]
    if case not in cases:
        cases.append(case)
    text = log.read_text(errors="replace")
    lines = text.splitlines()
    if not any(l.startswith("backend=") for l in lines):  # crashed, timed out or running
        results.setdefault((case, "(run)"), {})[mode] = "CRASH"
        reasons.setdefault(crash_reason(text), set()).add((mode, case))
        continue
    body = sections(lines)
    crash_logs = sorted(logdir.glob(f"{mode}_{case}.crash*.txt"))
    table = lines[next(i for i, l in enumerate(lines) if l.startswith("backend=")) + 2:]
    for row in table:
        r = row_re.match(row)
        if not r or row.startswith(" "):
            continue
        _, kernel, status, err, run = r.groups()
        section = body.get(kernel, [])
        if status.startswith("FAIL"):
            cell = "FAIL"
            if kernel == "(setup)":
                fail = next((l for l in lines if "setup/record FAILED" in l), status)
                detail = next((l for l in lines if "nested task error" in l), "")
                reason = detail or fail.split(":", 1)[-1]
            elif "crash" in status:  # the table truncates the status to 20 characters
                cell = "CRASH"
                reason = crash_reason(crash_logs[-1].read_text(errors="replace")) \
                    if crash_logs else "process crashed"
            else:
                reason = raise_reason(section, lines, kernel) or \
                    next((l for l in section if "error:" in l or "Reason:" in l),
                         next((l for l in section if "FAIL" in l), status))
                reason = reason.split("error:", 1)[-1] if "error:" in reason else reason
            reasons.setdefault(clean(reason), set()).add((mode, kernel))
        else:
            cell = SHORT.get(status, status) + f" {err}, {run}ms"
            gs = next((re.search(r"gather=(\d+) scatter=(\d+)", l) for l in section
                       if "gather=" in l), None)
            if gs and status.startswith("ok (raised)"):
                cell += f", {gs[1]}g/{gs[2]}s"
            if any("ref=cuda" in l for l in section):
                cell += " (ref CUDA)"
        results.setdefault((case, kernel), {})[mode] = cell

print("## Per case and kernel\n")
print("cell = status, rel. error vs. the KA CPU launch (native CUDA launch where marked), "
      "run time, gathers/scatters in the raised HLO; kept = kernel call not raised\n")
print("| case | kernel | " + " | ".join(MODES) + " |")
print("|---|---|" + "---|" * len(MODES))
for case in cases:
    for (c, k), cells in results.items():
        if c == case:
            row = [cells.get(mode, "–") for mode in MODES]
            print(f"| {c} | `{k}` | " + " | ".join(row) + " |")

print("\n## Per kernel (all cases)\n")
print("| kernel | " + " | ".join(MODES) + " |")
print("|---|" + "---|" * len(MODES))
for k in sorted({k for (_, k) in results}):
    row = []
    for mode in MODES:
        cells = [cells.get(mode) for (_, kk), cells in results.items() if kk == k]
        tags = sorted({x.split(" ")[0] for x in cells if x})
        row.append("/".join(tags) or "–")
    print(f"| `{k}` | " + " | ".join(row) + " |")

print("\n## Failure reasons\n")
for reason, where in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
    modes = sorted({m for m, _ in where})
    names = sorted({k for _, k in where})
    print(f"- {reason}  \n  ({', '.join(modes)}): " + ", ".join(f"`{n}`" for n in names))
