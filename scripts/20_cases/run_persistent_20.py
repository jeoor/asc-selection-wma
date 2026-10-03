"""Run the official 20 WMA cases with one model load and unchanged case settings."""

import argparse
import copy
import csv
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
from pytorch_lightning import seed_everything


ROOT = Path("/root/autodl-tmp/WMA20")
MODEL_DIR = ROOT / "unifolm-world-model-action"
CASES_DIR = ROOT / "ASC26-Embodied-World-Model-Optimization"


def official_number(script: Path, option: str) -> int:
    match = re.search(rf"--{option}\s+(\d+)", script.read_text())
    if match is None:
        raise ValueError(f"Missing {option} in {script}")
    return int(match.group(1))


def load_inference_module():
    sys.path.insert(0, str(MODEL_DIR / "scripts/evaluation"))
    source = Path(__file__).with_name("world_model_interaction.py")
    spec = importlib.util.spec_from_file_location("persistent_wma", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python run_persistent_20.py /absolute/result/directory")
    result_dir = Path(sys.argv[1])
    if not result_dir.is_absolute() or result_dir.exists():
        raise SystemExit(f"Result directory must be new and absolute: {result_dir}")
    result_dir.mkdir(parents=True)

    scripts = sorted(CASES_DIR.glob("unitree_*/case*/run_world_model_interaction.sh"))
    if len(scripts) != 20:
        raise RuntimeError(f"Expected 20 official cases, found {len(scripts)}")
    selected = os.environ.get("WMA_CASE_FILTER")
    if selected:
        requested = set(selected.split(","))
        scripts = [
            script for script in scripts
            if f"{script.parent.parent.name}/{script.parent.name}" in requested
        ]
        found = {f"{script.parent.parent.name}/{script.parent.name}" for script in scripts}
        if found != requested:
            raise RuntimeError(f"Unknown cases in filter: {sorted(requested - found)}")

    os.environ["HF_HOME"] = str(ROOT / "hf-cache")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = os.environ.get("WMA_CUDNN_BENCHMARK") == "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("WMA_ALLOW_TF32") == "1"
    os.chdir(MODEL_DIR)
    inference = load_inference_module()
    parser = inference.get_parser()
    common = parser.parse_args([
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
    summary_path = result_dir / "summary.csv"
    with summary_path.open("w", newline="") as file:
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
            full_fp32 = scenario == "unitree_z1_dual_arm_stackbox_v2" and case_name == "case1"
            args.fp32_decoder = False
            precision = "fp32" if full_fp32 else "fp16"
            args.amp_dtype = "none" if full_fp32 else "fp16"
            (case_result / "arguments.json").write_text(
                json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n"
            )
            (case_result / "precision.txt").write_text(precision + "\n")

            print(f"START {index}/20 {scenario}/{case_name} {precision}", flush=True)
            start = time.perf_counter()
            seed_everything(args.seed)
            amp_context = nullcontext() if full_fp32 else torch.autocast(
                device_type="cuda", dtype=torch.float16
            )
            with amp_context:
                cached = inference.run_inference(args, 1, 0, cached=cached)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            writer.writerow([scenario, case_name, n_iter, frame_stride, f"{elapsed:.6f}"])
            file.flush()
            print(f"DONE  {index}/20 {scenario}/{case_name} {elapsed:.3f}s", flush=True)

    print(f"COMPLETE {len(scripts)}/{len(scripts)}", flush=True)


if __name__ == "__main__":
    main()
