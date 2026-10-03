# Embodied World Model：20 Case 推理优化

姓名：乔可傲  
年级专业：24绿算  
大题：Embodied World Model

本仓库保存 20 个官方 Case 的 AMP Baseline、优化实验、源码补丁和结果记录。旧版单 Case 训练结果仍保留在 `results/` 的原目录中，不参与下面的 20 Case 判定。模型权重、输入数据和输出视频没有上传。

## 项目与环境

- [赛题仓库](https://github.com/ASC-Competition/ASC26-Embodied-World-Model-Optimization)
- [模型项目](https://github.com/unitreerobotics/unifolm-world-model-action)，代码提交号 `3e198de68de55f93f24b3ad623dd499390aaee45`
- [模型权重](https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual)：`ckpts/unifolm_wma_dual.ckpt`

20 个 Case 输入来自赛题仓库，按原场景和 Case 目录放在模型项目下。`configs/inference/world_model_interaction.yaml` 的 `data_dir` 按模型项目 README 设置为 `examples/world_model_interaction_prompts`。

| 项目 | 内容 |
| --- | --- |
| 机器来源 | 自备服务器（AutoDL，付费） |
| 操作系统 | Ubuntu 22.04.1 LTS |
| CPU、内存 | Intel Xeon Gold 6348（14 vCPU），120 GiB |
| GPU、驱动 | NVIDIA A800 80GB PCIe，590.48.01 |
| Python | 3.10.18 |
| PyTorch、CUDA | 2.3.1+cu121，PyTorch CUDA 12.1 |

## 20 Case 结果

| 实验 | 完整运行时间 | 总加速比 | 最低 PSNR | 结论 |
| --- | ---: | ---: | ---: | --- |
| AMP FP16 Baseline | 7683.18 s | 1.000× | 23.759502 dB | 20/20 完成；1 个 Case 低于 25 dB |
| 首次模型复用、异步写入，低分 Case 用 FP32 | 6257.32 s | 1.228× | 24.947893 dB | 时间和质量均未达标 |
| 恢复随机数状态，并对低分 Case 启用 TF32 矩阵运算 | 5913.84 s | 1.299× | 25.120085 dB | 20/20 完成，全部达标 |

新要求的总时间上限为 `7683.18 / 1.25 = 6146.544 s`。最终实验比 AMP Baseline 少用 1769.34 s，时间下降 23.03%。每个 Case 的 PSNR、轮数、帧数和帧间隔见 `results/20_cases/optimized/quality.csv`、`summary.csv` 与 `assessment.json`。

Baseline 每个 Case 单独启动 Python 并用 `/usr/bin/time` 计时，表中的 7683.18 s 是 20 次完整运行时间之和。中断的 Case 不计入汇总，恢复后重新完整运行。优化版一次启动后依次处理 20 个 Case，5913.84 s 是整个 Python 进程的墙钟时间。两组使用相同的 A800、驱动、项目基础提交号、权重和官方输入，但在不同日期运行，没有统计多次运行的波动。

## 做了什么

AMP FP16 只是本次对照基线，不计入优化。原运行方式每个 Case 都重新加载模型，推理过程中还会同步写逐轮视频。我尝试让 20 个 Case 共用一次模型加载，把逐轮视频交给后台线程写入，并让低分 Case 使用完整 FP32。首次完整测试加速 1.228×，但 `unitree_z1_dual_arm_stackbox_v2/case1` 的 PSNR 只有 24.947893 dB。

原因是独立运行时，设定种子后还会加载模型并消耗随机数；复用模型时，后续 Case 跳过加载，即使重新设相同种子，生成噪声也不同。修正后保存首次模型加载后的随机数状态，并在每个后续 Case 推理前恢复。修正后的该 Case 在完整 FP32 下生成了与独立 FP32 运行逐字节相同的视频。

该 Case 用 AMP FP16 时只有 23.759502 dB；完整 FP32 可以超过 25 dB，但耗时较长。最终实验全局允许 TF32 矩阵运算，但只让这个 Case 关闭 autocast；其余 19 个 Case 仍用 AMP FP16，最终视频与 Baseline 逐字节相同。TF32 定点测试为 25.120085 dB，20 Case 正式运行中该 Case 的视频与定点测试逐字节相同。[NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__tf32.html)说明 TF32 至少用 19 bit 表示数值；这里的开关是 `torch.backends.cuda.matmul.allow_tf32 = True`，没有使用低于 16-bit 的数据类型。

正式实验保持了官方的采样步数、交互轮数、输入和视频规格。20 个 Case 各自的帧数均与 AMP Baseline 相同，逐轮视频两组各有 392 个。其余 19 个 AMP FP16 Case 的最终 MP4 与 Baseline 逐字节相同。PSNR 使用赛题提供的 `psnr_score_for_challenge.py` 计算。

## 运行与复核

两个补丁都针对上面的模型项目提交号，从干净源码分别应用，不叠加：

- `patches/wma_optimizations.patch`：提供 AMP 开关；Baseline 只使用 `--amp_dtype fp16`。
- `patches/wma_20case_optimizations.patch`：包含 AMP 参数、模型复用、异步写入和随机数状态恢复，是最终优化版使用的源码。

本次 Baseline 的启动命令为：

```bash
bash /root/autodl-tmp/WMA20/run_amp_baseline_20.sh \
  /root/autodl-tmp/WMA20/amp_baseline_20_20260926_04
```

仓库中的相同脚本位于 `scripts/20_cases/run_amp_baseline_20.sh`。该脚本逐一读取官方 20 个 Case 的轮数和帧间隔，使用原有的 50 步 DDIM 与其他输入参数。每个 Case 的命令和完整日志保存在 `results/20_cases/amp_baseline/`。

优化版的启动命令为：

```bash
WMA_ALLOW_TF32=1 \
WMA_OPT_DIR=/root/autodl-tmp/WMA20/persistent_rng_tf32_20_20261003_01 \
WMA_TOOLS_DIR=/root/autodl-tmp/WMA20/optimized_eval_v5 \
WMA_RUN_TAG=persistent_rng_tf32_20_20261003_01 \
bash /root/autodl-tmp/WMA20/optimized_eval_v5/launch_after_baseline.sh
```

`launch_after_baseline.sh` 用 `/usr/bin/time` 计时，并在推理后调用 `score_persistent_20.sh` 和 `assess_20.py`。服务器上的脚本目录是 `/root/autodl-tmp/WMA20/optimized_eval_v5`；仓库中相同脚本位于 `scripts/20_cases/`，其 SHA-256 与本次运行脚本相同。复跑时需要把脚本中的 `ROOT` 和启动命令的 `WMA_TOOLS_DIR` 指向自己的目录，并使用新的结果目录，脚本不会覆盖已有结果。

`results/20_cases/` 分别保存 Baseline、未达标尝试和最终实验的时间、PSNR、视频信息及日志。`optimized/assessment.json` 会核对 20 个 Case、每个 Case 的帧数与 PSNR，以及总加速比。仓库没有上传模型权重、原始视频和生成视频。
