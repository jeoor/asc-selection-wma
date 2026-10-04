"""Check the single-variable ablations against original video bytes and cases.

The reload-only experiment is deliberately marked segmented after an outage;
its case-time sum is not presented as a single-process end-to-end measurement.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def keyed(data):
    result = {(row["scenario"], row["case"]): row for row in data}
    if len(result) != len(data):
        raise ValueError("Duplicate case in summary")
    return result


def video(root, key):
    matches = list((root / key[0] / key[1] / "output/inference").glob("*_full_fs*.mp4"))
    if len(matches) != 1:
        raise ValueError(f"Expected one final video for {key} under {root}: {matches}")
    return matches[0]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(sys.argv[1])
    modes = {
        "cached_sync": root / "full_cached_sync",
        "cached_async": root / "full_cached_async",
        "reload_first10": root / "full_reload_async",
        "reload_tail10": root / "reload_async_tail10",
    }
    environments = {
        name: json.loads((path / "environment.json").read_text())
        for name, path in modes.items()
    }
    control = environments["cached_async"]
    missing_tf32_metadata = []
    for name, environment in environments.items():
        for field in ("source_sha256", "torch", "torch_cuda", "gpu", "nvidia_smi"):
            if environment[field] != control[field]:
                raise ValueError(f"Environment mismatch in {name}: {field}")
        if "allow_tf32" not in environment:
            missing_tf32_metadata.append(name)
        elif environment["allow_tf32"] != control["allow_tf32"]:
            raise ValueError(f"Environment mismatch in {name}: allow_tf32")
    summaries = {name: keyed(rows(path / "summary.csv")) for name, path in modes.items()}
    first, tail = summaries["reload_first10"], summaries["reload_tail10"]
    if len(first) != 10 or len(tail) != 10 or first.keys() & tail.keys():
        raise ValueError("Reload segments must have ten disjoint cases each")
    reload_rows = first | tail
    keys = set(summaries["cached_async"])
    if len(keys) != 20 or set(summaries["cached_sync"]) != keys or set(reload_rows) != keys:
        raise ValueError("Ablation case sets do not match all twenty official cases")

    per_case = []
    mismatches = []
    for key in sorted(keys):
        row = {"scenario": key[0], "case": key[1]}
        for name, data in (*summaries.items(), ("reload", reload_rows)):
            if key not in data:
                continue
            current = data[key]
            for field in ("n_iter", "frame_stride"):
                if current[field] != summaries["cached_async"][key][field]:
                    raise ValueError(f"Parameter mismatch in {name}: {key} {field}")
            row[name + "_seconds"] = float(current["real_seconds"])
        control_args = json.loads((modes["cached_async"] / key[0] / key[1] / "arguments.json").read_text())
        control_args.pop("savedir")
        for name in ("cached_sync", "reload_first10" if key in first else "reload_tail10"):
            args = json.loads((modes[name] / key[0] / key[1] / "arguments.json").read_text())
            args.pop("savedir")
            if args != control_args:
                raise ValueError(f"Arguments differ in {name}: {key}")
        async_digest = digest(video(modes["cached_async"], key))
        sync_matched = digest(video(modes["cached_sync"], key)) == async_digest
        row["sync_video_matches_async"] = sync_matched
        if not sync_matched:
            mismatches.append("sync:" + "/".join(key))
        segment = "reload_first10" if key in first else "reload_tail10"
        row["reload_segment"] = segment
        matched = digest(video(modes[segment], key)) == async_digest
        row["reload_video_matches_cached_async"] = matched
        if not matched:
            mismatches.append("/".join(key))
        per_case.append(row)

    if mismatches:
        raise ValueError(f"Ablation videos differ; score them separately: {mismatches}")
    quality = keyed(rows(modes["cached_async"] / "quality.csv"))
    if set(quality) != keys:
        raise ValueError("Cached asynchronous quality must include all cases")

    totals = {
        "cached_sync_case_seconds": sum(row["cached_sync_seconds"] for row in per_case),
        "cached_async_case_seconds": sum(row["cached_async_seconds"] for row in per_case),
        "reload_segmented_case_seconds": sum(row["reload_seconds"] for row in per_case),
        "cached_sync_wall_seconds": float((modes["cached_sync"] / "total_time.txt").read_text().split("real ")[1].splitlines()[0]),
        "cached_async_wall_seconds": float((modes["cached_async"] / "total_time.txt").read_text().split("real ")[1].splitlines()[0]),
    }
    totals["async_vs_sync_wall_saved_seconds"] = (
        totals["cached_sync_wall_seconds"] - totals["cached_async_wall_seconds"]
    )
    totals["reload_vs_cached_async_case_saved_seconds"] = (
        totals["reload_segmented_case_seconds"] - totals["cached_async_case_seconds"]
    )
    totals["reload_segments_are_not_one_continuous_end_to_end_run"] = True
    totals["all_twenty_sync_videos_match_scored_cached_async_videos"] = True
    totals["all_twenty_reload_videos_match_scored_cached_async_videos"] = True
    totals["minimum_psnr_db"] = min(float(row["psnr"]) for row in quality.values())
    totals["missing_tf32_environment_metadata"] = missing_tf32_metadata

    output = root / "ablation_assessment.json"
    output.write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n")
    with (root / "ablation_case_times.csv").open("w", newline="") as file:
        fields = list(dict.fromkeys(field for row in per_case for field in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_case)
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
