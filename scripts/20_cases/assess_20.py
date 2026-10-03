"""Check the revised 20-case timing and quality requirements."""

import csv
import json
from pathlib import Path
import sys


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    result = {(row["scenario"], row["case"]): row for row in rows}
    if len(rows) != 20 or len(result) != 20:
        raise ValueError(f"Expected 20 distinct cases in {path}, got {len(rows)}")
    return result


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python assess_20.py BASELINE_DIR OPTIMIZED_DIR")
    baseline_dir, optimized_dir = (Path(arg) for arg in sys.argv[1:])
    baseline = read_rows(baseline_dir / "summary.csv")
    optimized = read_rows(optimized_dir / "summary.csv")
    quality = read_rows(optimized_dir / "quality.csv")
    if baseline.keys() != optimized.keys() or baseline.keys() != quality.keys():
        raise ValueError("Baseline, optimized run, and quality files do not have identical cases")

    problems = []
    for key in sorted(baseline):
        b, o, q = baseline[key], optimized[key], quality[key]
        for field in ("n_iter", "frame_stride"):
            if b[field] != o[field] or b[field] != q[field]:
                problems.append(f"{key}: {field} differs")
        if int(q["frames"]) != int(b["frames"]):
            problems.append(f"{key}: frame count differs")
        if float(q["psnr"]) < 25.0:
            problems.append(f"{key}: PSNR {q['psnr']} dB < 25 dB")

    baseline_seconds = sum(float(row["real_seconds"]) for row in baseline.values())
    with (optimized_dir / "total_time.txt").open() as file:
        real_lines = [line for line in file if line.startswith("real ")]
    if len(real_lines) != 1:
        raise ValueError("Missing total wall time for optimized run")
    optimized_seconds = float(real_lines[0].split()[1])
    speedup = baseline_seconds / optimized_seconds
    if speedup < 1.25:
        problems.append(f"Total speedup {speedup:.4f}x < 1.25x")

    result = {
        "baseline_20_total_seconds": baseline_seconds,
        "optimized_20_total_seconds": optimized_seconds,
        "speedup": speedup,
        "lowest_optimized_psnr_db": min(float(row["psnr"]) for row in quality.values()),
        "case_count": 20,
        "passes_revised_requirements": not problems,
        "problems": problems,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (optimized_dir / "assessment.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
