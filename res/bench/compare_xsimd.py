#!/usr/bin/env python3
"""Repeatedly runs the already-built xsimd/mkn.avx comparison binaries and averages.

test/compare_xsimd.cpp and test/compare_mkn_avx.cpp are independent binaries
(separate processes) so neither implementation's measurement is biased by
running first/second in the same process. A single run of either is still
too noisy on its own (turbo ramp-up, allocator/page state, scheduler jitter),
so this drives each compiled binary as a fresh OS process, several times,
and averages across runs.

Build the binaries first, e.g.:
    mkn clean build run -p xsimd -Otda "-std=c++20" -l -pthread -g 0
    mkn clean build run -p compare_mkn_avx -Otda "-std=c++20" -l -pthread -g 0

Usage:
    python3 res/bench/compare_xsimd.py [--runs N]
        [--xsimd-bin PATH] [--mkn-avx-bin PATH]
"""

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

TIME_RE = re.compile(r"Time taken:\s*([0-9.eE+-]+)")


def repo_root():
    return Path(__file__).resolve().parents[2]


def run_once(bin_path):
    out = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True).stdout
    times = [float(m.group(1)) for m in TIME_RE.finditer(out)]
    if len(times) != 1:
        raise RuntimeError(f"expected exactly one 'Time taken:' line, got {len(times)}:\n{out}")
    return times[0]


def collect(label, bin_path, runs):
    if not bin_path.is_file():
        sys.exit(
            f"binary not found: {bin_path}\n"
            "build it first, see this script's module docstring"
        )
    samples = []
    for i in range(runs):
        t = run_once(bin_path)
        samples.append(t)
        print(f"{label} run {i + 1}/{runs}: {t:.6f}")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="number of process invocations")
    parser.add_argument("--xsimd-bin", type=Path, default=None)
    parser.add_argument("--mkn-avx-bin", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    binaries = {
        "xsimd": args.xsimd_bin or root / "bin" / "xsimd" / "mkn.avx",
        "mkn.avx": args.mkn_avx_bin or root / "bin" / "compare_mkn_avx" / "mkn.avx",
    }

    samples = {}
    for label, bin_path in binaries.items():
        samples[label] = collect(label, bin_path, args.runs)

    print()
    print(f"{'impl':<10}{'mean':>12}{'median':>12}{'stdev':>12}")
    means = {}
    for label, vals in samples.items():
        mean = statistics.mean(vals)
        means[label] = mean
        median = statistics.median(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{label:<10}{mean:>12.6f}{median:>12.6f}{stdev:>12.6f}")

    print()
    ratio = means["mkn.avx"] / means["xsimd"]
    faster = "xsimd" if ratio > 1 else "mkn.avx"
    print(f"mkn.avx/xsimd mean ratio: {ratio:.3f}  ({faster} faster)")


if __name__ == "__main__":
    main()
