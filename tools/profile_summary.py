#!/usr/bin/env python3
"""Summarise `Reactant.@profile` logs written by the E7/E9 `profile` tasks.

For every `== profile ...` block, read the FRAMEWORK OP STATISTICS table (needs a wide
terminal when the log was written: COLUMNS=400 LINES=2000) and print, per kernel category,
the device time and the bytes moved per rhs! call (self time x memory bandwidth), plus the
number of passes over the state that this traffic corresponds to.

usage: tools/profile_summary.py <log> [state_bytes] [hlo.txt ...]
  state_bytes: bytes of one state (all variables), e.g. 16.8M nodes x 4 vars x 4 B = 268435456
  hlo.txt: optimized XLA HLO dumps (`@code_xla`), one per profile block in order; TPU kernels
           are named fusion.N, so they are classified by the ops inside the fused computation
"""
import re
import sys

CATEGORIES = [  # first match wins
    ("matrix product", re.compile(r"gemm|dot|convolution|cublas|matmul", re.I)),
    ("transpose", re.compile(r"transpose|copy", re.I)),
    ("concatenate", re.compile(r"concatenate", re.I)),
    ("gather/scatter", re.compile(r"gather|scatter", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"fusion|loop|multiply|add|subtract|divide", re.I)),
]


def category(name, hlo=None):
    if hlo is not None and name in hlo:
        ops = hlo[name]
        if ops & {"convolution", "dot"}:
            return "matrix product"
        if ops & {"sine", "cosine"}:
            return "sin/cos"
        for op, cat in (("gather", "gather/scatter"), ("scatter", "gather/scatter"),
                        ("concatenate", "concatenate"), ("reduce", "reduce"),
                        ("transpose", "transpose")):
            if op in ops:
                return cat
        if ops <= {"copy", "copy-start", "copy-done", "bitcast", "parameter", "tuple",
                   "get-tuple-element"}:
            return "transpose"   # layout-changing copies
        return "elementwise"
    for cat, pat in CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def read_hlo(path):
    """Map each instruction name to the set of ops it executes, following `calls=` into
    fused computations (also nested ones)."""
    comp_ops, comp_callees, inst_callee, inst_op, cur = {}, {}, {}, {}, None
    inst = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = .*? ([a-z][a-z\-]*)\(")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(?:ENTRY )?%([\w.\-]+) ", line)
        if m and line.rstrip().endswith("{"):
            cur = m.group(1)
            comp_ops[cur], comp_callees[cur] = set(), set()
            continue
        m = inst.match(line)
        if m and cur is not None:
            name, op = m.group(1), m.group(2)
            comp_ops[cur].add(op)
            inst_op[name] = op
            c = re.search(r"calls=%([\w.\-]+)", line)
            if c:
                inst_callee[name] = c.group(1)
                comp_callees[cur].add(c.group(1))

    def expand(comp, seen):
        if comp in seen:
            return set()
        seen.add(comp)
        ops = set(comp_ops.get(comp, ()))
        for callee in comp_callees.get(comp, ()):
            ops |= expand(callee, seen)
        return ops

    return {name: {op} | (expand(inst_callee[name], set()) if name in inst_callee else set())
            for name, op in inst_op.items()}


def cells(line):
    return [c.strip() for c in line.strip().strip("│").split("│")]


def summarise(block_title, rows, nrepeat, state_bytes, hlo=None):
    per_cat = {}
    for name, secs, bw in rows:
        c = per_cat.setdefault(category(name, hlo), [0.0, 0.0, 0])
        c[0] += secs / nrepeat
        c[1] += secs * bw * 1e9 / nrepeat
        c[2] += 1
    total_t = sum(v[0] for v in per_cat.values())
    total_b = sum(v[1] for v in per_cat.values())
    print(f"\n{block_title}")
    print(f"  {'category':<16} {'ms/rhs':>8} {'share':>6} {'GB/rhs':>8} {'GB/s':>7} {'ops':>4}")
    for cat, (t, b, n) in sorted(per_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"  {cat:<16} {t * 1e3:8.3f} {t / total_t:6.1%} {b / 1e9:8.3f} {b / t / 1e9 if t else 0:7.0f} {n:4d}")
    line = f"  {'total':<16} {total_t * 1e3:8.3f} {'':>6} {total_b / 1e9:8.3f} {total_b / total_t / 1e9:7.0f}"
    if state_bytes:
        line += f"   = {total_b / state_bytes:.1f} state-sized passes"
    print(line)


def main():
    path = sys.argv[1]
    state_bytes = float(sys.argv[2]) if len(sys.argv) > 2 else None
    hlos = [read_hlo(h) for h in sys.argv[3:]]
    title, rows, in_fw, header = None, [], False, None
    nrepeat, block = 5, -1
    hlo = lambda: hlos[block] if 0 <= block < len(hlos) else None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("== profile"):
            if title and rows:
                summarise(title, rows, nrepeat, state_bytes, hlo())
            title, rows, in_fw, header = line.strip(), [], False, None
            block += 1
        elif "FRAMEWORK OP STATISTICS" in line:
            in_fw = True
        elif "SUMMARY" in line:
            in_fw = False
        elif in_fw and line.startswith("│"):
            c = cells(line)
            if c[0] == "Operation":
                header = c
                continue
            if header is None or len(c) != len(header):
                continue
            rec = dict(zip(header, c))
            if rec.get("Host/Device") != "Device":
                continue
            try:
                secs = float(rec["Total Self-Time (s)"])
                bw = float(rec.get("Memory BW (GB/s)", "0") or 0)
            except ValueError:
                continue
            rows.append((rec["Operation"], secs, bw))
    if title and rows:
        summarise(title, rows, nrepeat, state_bytes, hlo())


if __name__ == "__main__":
    main()
