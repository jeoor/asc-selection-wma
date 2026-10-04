"""Estimate 20-case loop and non-loop time from saved tqdm logs.

tqdm rounds its displayed elapsed time to seconds, so all loop figures here
are approximate. This does not identify individual operation costs.
"""

import csv
import json
from pathlib import Path
import re
import sys


PROGRESS = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)<00:00")
START = re.compile(r"START \d+/20 ([^\s]+)")


def final_loop_seconds(text: str, expected_iterations: int) -> int:
    matches = PROGRESS.findall(text)
    if not matches:
        raise ValueError("No finished tqdm line")
    done, total, minutes, seconds = matches[-1]
    if int(done) != expected_iterations or int(total) != expected_iterations:
        raise ValueError(f"Unexpected iteration count: {done}/{total}")
    return 60 * int(minutes) + int(seconds)


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    result = {f"{row['scenario']}/{row['case']}": row for row in rows}
    if len(rows) != 20 or len(result) != 20:
        raise ValueError(f"Expected 20 distinct cases in {path}")
    return result


def optimized_loop_times(console: Path, rows: dict[str, dict[str, str]]) -> dict[str, int]:
    chunks = re.split(r"(?=START \d+/20 )", console.read_text(errors="replace"))
    result = {}
    for chunk in chunks:
        start = START.search(chunk)
        if start is None:
            continue
        case = start.group(1)
        if case not in rows or case in result:
            raise ValueError(f"Unknown or repeated case: {case}")
        if not re.search(rf"DONE\s+\d+/20\s+{re.escape(case)}\s", chunk):
            raise ValueError(f"Case did not finish: {case}")
        result[case] = final_loop_seconds(chunk, int(rows[case]["n_iter"]))
    if result.keys() != rows.keys():
        raise ValueError("Optimized log does not cover the same 20 cases")
    return result


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize_loop_timing.py BASELINE_DIR OPTIMIZED_DIR OUTPUT_JSON")
    baseline_dir, optimized_dir, output = map(Path, sys.argv[1:])
    baseline = read_csv(baseline_dir / "summary.csv")
    optimized = read_csv(optimized_dir / "summary.csv")
    if baseline.keys() != optimized.keys():
        raise ValueError("Baseline and optimized cases differ")

    baseline_loops = {
        case: final_loop_seconds(
            (baseline_dir / row["scenario"] / row["case"] / "run.log").read_text(errors="replace"),
            int(row["n_iter"]),
        )
        for case, row in baseline.items()
    }
    optimized_loops = optimized_loop_times(optimized_dir / "console.log", optimized)
    baseline_total = sum(float(row["real_seconds"]) for row in baseline.values())
    with (optimized_dir / "total_time.txt").open() as file:
        optimized_total = float(next(line.split()[1] for line in file if line.startswith("real ")))
    baseline_loop_total = sum(baseline_loops.values())
    optimized_loop_total = sum(optimized_loops.values())
    result = {
        "method": "Sum each case's rounded final tqdm elapsed display; non-loop = wall time - loop sum",
        "baseline_wall_seconds": round(baseline_total, 2),
        "baseline_loop_approx_seconds": baseline_loop_total,
        "baseline_non_loop_approx_seconds": round(baseline_total - baseline_loop_total, 2),
        "optimized_wall_seconds": optimized_total,
        "optimized_loop_approx_seconds": optimized_loop_total,
        "optimized_non_loop_approx_seconds": round(optimized_total - optimized_loop_total, 2),
        "loop_reduction_approx_seconds": baseline_loop_total - optimized_loop_total,
        "non_loop_reduction_approx_seconds": round(
            (baseline_total - baseline_loop_total) - (optimized_total - optimized_loop_total), 2
        ),
        "case_count": 20,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
