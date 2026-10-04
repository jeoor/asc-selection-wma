"""Measure model reuse and asynchronous video writing separately.

This is a diagnostic runner. It keeps the official case arguments and the
final experiment's precision policy; only the selected engineering switch
changes between runs.
"""

import argparse
import copy
import csv
from concurrent.futures import Future
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# huggingface_hub reads these variables during import, before main() runs.
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/WMA20/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from pytorch_lightning import seed_everything


ROOT = Path("/root/autodl-tmp/WMA20")
MODEL_DIR = ROOT / "unifolm-world-model-action"
CASES_DIR = ROOT / "ASC26-Embodied-World-Model-Optimization"


class InlineExecutor:
    """Run the unchanged video-writing function in the caller thread."""

    def __init__(self, max_workers=1):
        if max_workers != 1:
            raise ValueError("Expected one video writer")

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True):
        pass


def official_number(script: Path, option: str) -> int:
    match = re.search(rf"--{option}\s+(\d+)", script.read_text())
    if match is None:
        raise ValueError(f"Missing {option} in {script}")
    return int(match.group(1))


def load_inference_module():
    sys.path.insert(0, str(MODEL_DIR / "scripts/evaluation"))
    source = Path(__file__).with_name("world_model_interaction.py")
    spec = importlib.util.spec_from_file_location("ablation_wma", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cached_async", "cached_sync", "reload_async"))
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument("--disable-tf32", action="store_true")
    options = parser.parse_args()

    result_dir = options.result_dir
    if not result_dir.is_absolute() or result_dir.exists():
        raise SystemExit(f"Result directory must be new and absolute: {result_dir}")

    scripts = sorted(CASES_DIR.glob("unitree_*/case*/run_world_model_interaction.sh"))
    if len(scripts) != 20:
        raise RuntimeError(f"Expected 20 official cases, found {len(scripts)}")
    if options.cases:
        wanted = set(options.cases)
        scripts = [
            script for script in scripts
            if f"{script.parent.parent.name}/{script.parent.name}" in wanted
        ]
        found = {f"{script.parent.parent.name}/{script.parent.name}" for script in scripts}
        if found != wanted:
            raise ValueError(f"Unknown cases: {sorted(wanted - found)}")

    result_dir.mkdir(parents=True)
    os.environ["HF_HOME"] = str(ROOT / "hf-cache")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = not options.disable_tf32
    os.chdir(MODEL_DIR)
    inference, source = load_inference_module()
    if options.mode == "cached_sync":
        inference.ThreadPoolExecutor = InlineExecutor

    environment = {
        "mode": options.mode,
        "cases": [f"{s.parent.parent.name}/{s.parent.name}" for s in scripts],
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    (result_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n"
    )

    common = inference.get_parser().parse_args([
        "--seed", "123",
        "--ckpt_path", "ckpts/unifolm_wma_dual.ckpt",
        "--config", "configs/inference/world_model_interaction.yaml",
        "--savedir", str(result_dir),
        "--bs", "1", "--height", "320", "--width", "512",
        "--unconditional_guidance_scale", "1.0",
        "--ddim_steps", "50", "--ddim_eta", "1.0",
        "--prompt_dir", "unitree_g1_pack_camera/case1/world_model_interaction_prompts",
        "--dataset", "unitree_g1_pack_camera",
        "--video_length", "16", "--frame_stride", "6",
        "--n_action_steps", "16", "--exe_steps", "16", "--n_iter", "11",
        "--timestep_spacing", "uniform_trailing",
        "--guidance_rescale", "0.7", "--perframe_ae",
        "--amp_dtype", "fp16",
    ])

    cached = None
    with (result_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scenario", "case", "n_iter", "frame_stride", "real_seconds"])
        file.flush()

        for index, script in enumerate(scripts, start=1):
            scenario = script.parent.parent.name
            case_name = script.parent.name
            n_iter = official_number(script, "n_iter")
            frame_stride = official_number(script, "frame_stride")
            case_result = result_dir / scenario / case_name
            case_result.mkdir(parents=True)

            args = copy.copy(common)
            args.savedir = str(case_result / "output")
            args.prompt_dir = f"{scenario}/{case_name}/world_model_interaction_prompts"
            args.dataset = scenario
            args.frame_stride = [frame_stride]
            args.n_iter = n_iter
            args.fp32_decoder = False
            low_quality_case = scenario == "unitree_z1_dual_arm_stackbox_v2" and case_name == "case1"
            args.amp_dtype = "none" if low_quality_case else "fp16"
            (case_result / "arguments.json").write_text(
                json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n"
            )

            if options.mode == "reload_async" and cached is not None:
                cached = None
                gc.collect()
                torch.cuda.empty_cache()

            print(f"START {index}/{len(scripts)} {scenario}/{case_name}", flush=True)
            start = time.perf_counter()
            seed_everything(args.seed)
            amp = nullcontext() if low_quality_case else torch.autocast(
                device_type="cuda", dtype=torch.float16
            )
            with amp:
                cached = inference.run_inference(args, 1, 0, cached=cached)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            writer.writerow([scenario, case_name, n_iter, frame_stride, f"{elapsed:.6f}"])
            file.flush()
            print(f"DONE {index}/{len(scripts)} {scenario}/{case_name} {elapsed:.3f}s", flush=True)

    print(f"COMPLETE {len(scripts)}/{len(scripts)}", flush=True)


if __name__ == "__main__":
    main()
