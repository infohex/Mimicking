[English](release-2.0.md) | 中文

# Mimicking 2.0 正式版（2026-11-01）

Mimicking 2.0 是第一个**全部用我们自己的生成器权重**跑实时变声的版本：内容编码（XLS-R L12，Apache-2.0）→ K=4096 软分配内容码本 → 长度调节器 → DiT 流匹配 → HiFT 声码器（Apache-2.0），调节器与 DiT 为 MimickingTrain 自写定义、自训权重。
在 32 对 LibriTTS-R 评测上，2.0 的英文说话人模型像目标 0.712 / 像源 0.419（流式），流式 WER 7.6 %；实时链路在 160 ms 块下端到端约 330 ms。

本文是"冻结清单 + 最终参数 + 数字 + 已知限制"。接入方法见 `docs/integration-guide.md`；训练与诊断过程中踩过的坑见 MimickingTrain `docs/lessons-learned.md`。

## 1. 正式版由什么构成

### 权重（MimickingTrain `runs/Mimicking-v2.0/`，不进 git，随发布一起归档）

| 文件 | 内容 | sha256（前 16 位） |
|---|---|---|
| `ema.safetensors` | 2.0 生成器 EMA 权重，一个文件 108 个张量：`dit.*` 88 个（DiT，18.27 M 参数）+ `regulator.*` 20 个（调节器，2.32 M），合计 20.6 M | `8d29af773a441995` |
| `xlsr12_k4096.safetensors` | 内容码本：4096 × 1024 质心 + tau（**推理必需**，导出时烘进图） | `f97ef495a7b56103` |
| `regulator_norm_stats.json` | 调节器 4 个 GroupNorm 的整句统计量（64 句均值，**导出必需**） | `b70e30f31ad1f596` |
| `config.json` | 微调配置（见 §2） | `05f68988458a3b45` |
| `step_0010000.pt` | 训练器检查点（模型 + 优化器 + EMA），续训用 | `7ac20d9fc4875ec1` |

### 导出的模型套（`Mimicking/models/`，manifest `name` = `Mimicking-v2.0`，`version` = `2.0.1`）

| 目录 | 块长 | 右前瞻 | 用途 |
|---|---|---|---|
| `Mimicking-v2.0-lowlat` | 240 ms（配置 0.25 s，按 20 ms 格取整） | 0.02 s | **默认套**，Lab 启动即加载 |
| `Mimicking-v2.0-b0.16-r0.02` | 160 ms | 0.02 s | 延时优先；GPU 占用约 19 % |
| `Mimicking-v2.0-b0.32-r0.02` | 320 ms | 0.02 s | 算力优先；延时多 80 ms |

每套 10 个文件：`xlsr.onnx`、`xlsr_ref.onnx`、`campplus.onnx`、`hift_*`（Apache-2.0 模型的图；由 `mimictrain.export.shared` 直接从 `ref/models` 的权重导出）+ 我们导出的 `length_regulator.onnx`、`length_regulator_ref.onnx`（fp32，码本与冻结统计量已烘入）、`dit_step.onnx`、`dit_step_fused.onnx`（fp16 + ORT 融合算子）。
manifest 多了两个字段：`content_codebook`（码本文件、sha256、K、tau、烘入了哪两张图）和 `regulator_norm_stats`（统计量来源与数值）。**版本号 2.0.1 = 2.0 权重 + 冻结的调节器统计量**；2.0.0 的图在流式下会糊（见 §5 第 1 条），已全部就地重导。

### 语音门

`models/vad/silero_vad.onnx`（Silero VAD v6，MIT，1.2 MB，sha256 前 16 位 `7ed98ddbad84ccac`，许可文本在同目录）。`mimic_onnx::SileroVad` 实现 `Vad` 接口，CPU 上每 240 ms 块约 1 ms。

## 2. 最终参数

### 模型

| 级 | 参数 |
|---|---|
| 内容编码 | `facebook/wav2vec2-xls-r-300m` 第 12 层，1024 维，50 Hz，输入 16 kHz，逐窗均值方差归一（图内） |
| 内容码本 | k-means K=4096（`tools/content_gate.py fit`，训练集 XLS-R 帧），软分配 softmax(−d²/tau)，tau = 训练帧的平均最近平方距离，tau_scale 1.0 |
| 长度调节器 | Linear 1024→384，最近邻插值到 86.13 Hz 梅尔帧率，4 × (Conv3 + GroupNorm(1,384) + Mish)，1×1 Conv；推理时 GroupNorm 用冻结常数 |
| DiT | 384 维、9 层、6 头、RoPE、SwiGLU、U-ViT 跳连、时间与音色 token；流匹配，推理 10 步欧拉 + CFG 0.7 |
| 风格编码 | CAM++ `campplus_cn_common`（192 维；2.x 计划换英文模型） |
| 声码器 | HiFT（CosyVoice-300M），80 维梅尔 @ 22.05 kHz，hop 256 |

### 训练（MimickingTrain）

| 阶段 | 设置 |
|---|---|
| 1.0（`runs/Mimicking-v1.0`，运行名 tiny_sp） | 从零训练 100 k 步；LibriTTS-R train-clean-100 / 360 / other-500；验证 dev-clean |
| 2.0 微调（`configs/ft_vq4096.json`） | 从 1.0 EMA 热启动，码本插入后微调 10 000 步；两张 A4000（DDP，gloo），每卡每步 24 000 帧；lr 5e-5，warmup 500，余弦到 0.2 倍；AdamW β (0.9, 0.98)，wd 0.01，梯度裁剪 1.0；EMA 0.9999；提示 3 s，无提示概率 0.1；句长 90–1300 帧；输入扰动：音高 ±8 半音（≥2）、共振峰 ±35 %（≥12 %）、3 段 EQ ±12 dB、增益 ±6 dB；4 h 06 min |
| 冻结点 | 第 10 000 步（相似度在 2 500 步后已平台，续训到 30 k 的尝试在 11 400 步停止） |

### 导出（`python -m mimictrain.export`）

`--block 0.25 --right 0.02`（另 0.16 / 0.32）`--prompt 3.0`，`--train-config`（烘码本）`--norm-stats`（冻结统计量）`--shared from-weights`（默认：XLS-R / CAM++ / HiFT 从 Apache 权重导出，夹具写到套内 `fixtures/`）`--version 2.0.1`。交叉淡化 0.05 s、左上下文 2.5 s（XLS-R）/ 0.5 s（DiT）为几何默认值。对拍（`tools/export_parity.py`）：调节器 fp32 6.5e-4、CUDA TF32 2.7e-3；DiT 融合图 3.4e-4。

### 运行时（Lab 实测的推荐值）

| 项 | 值 | 说明 |
|---|---|---|
| 块长 | 160 或 240 ms | 每块计算 29–30 ms 与块长无关；160 ms 时 GPU 约 19 %，240 ms 约 12 % |
| 右前瞻 | 0.02 s | 最小格；放大到 0.1 s 对 2.0 无益 |
| 去噪步数 / CFG | 10 / 0.7 | |
| 门控 | Silero VAD，阈值 0.5，关门需连续 2 块低于 0.35，绝对地板 −60 dBFS | 能量门仅作缺模型时的退路；离线文件转换不开门 |
| 输出余量 | 50 ms（GPU 锁频后）/ 150 ms（不锁频） | 锁频：`nvidia-smi -i N -lgc 1800,1800`，p95−p50 从 60 ms 降到 2 ms |
| 队列控制 | 播放速率微调 ±1 %（P：40 ms 偏差 4 s 内拉回；慢积分），补静音 / 丢输入仅兜底 | 吸收麦克风与扬声器 0.1–0.3 % 的时钟漂移 |
| 麦克风电平 | 说话时 −25 dBFS 左右，关闭系统的"音频增强 / 自动增益" | −44 dBFS 偏低，且 AGC 会在响的敲击后压低几秒 |
| 延时 | 160 ms 块：软件链路 234 ms（块 160 + 计算 33 + 余量 41）+ 算法固有 80 ms + 声卡 20–40 ms ≈ 330–350 ms；240 ms 块约 +80 ms | 算法固有 = 右前瞻 20 + 交叉淡化 40 + SOLA 搜索 20 |

## 3. 数字

### 离线整句（32 对 LibriTTS-R，3 s 提示；英文 CAM++ = VoxCeleb 模型，中文 = campplus_cn_common）

| 版本 | 英文 像目标 / 像源 | 中文 像目标 / 像源 | WER（源 6.3 %） |
|---|---|---|---|
| 1.0 | 0.635 / 0.524 | 0.842 / 0.735 | — |
| 2.0（图 2.0.0） | 0.700 / 0.432 | 0.870 / 0.688 | 6.7 % |
| 2.0（图 2.0.1，冻结统计量） | 0.708 / 0.439 | 0.874 / 0.690 | 7.2 % |

### 流式（MimickingLab 无界面文件模式，同 32 对，块 240 ms，右前瞻 0.02 s，无门控）

| 套 | WER | 英文 像目标 / 像源 / 超地板 | 中文 像目标 / 超地板 |
|---|---|---|---|
| 2.0 图 2.0.0 | 16.1 % | 0.707 / 0.394 / +0.113 | 0.878 / +0.070 |
| **2.0 图 2.0.1（正式版）** | **7.6 %** | 0.712 / 0.419 / +0.138 | 0.875 / +0.069 |

块长 160 / 240 / 320 ms 的 4 对对比：相似度与 WER 在噪声内无差别。

### 噪声（BO 的真实录音，240 ms 块开门数；无门控时两套模型都满音量胡说）

| 录音 | 能量门 | Silero 门 |
|---|---|---|
| 键盘（34.8 s） | 114 / 144 | 0 / 144 |
| 敲桌（43.1 s） | 152 / 179 | 0 / 179 |
| 咳嗽喷嚏（15 s） | 44 / 62 | 7 / 62 |
| 办公室背景含远处人声（34 s） | 133 / 141 | 40 / 141（人声按设计放行） |
| 干净语音（召回） | 20 / 32 | 32 / 32 |

## 4. 权重与接口（给替换者）

- 权重：调节器与 DiT 是我们的（1.0 从零训练 + 2.0 码本微调）；XLS-R、CAM++、HiFT 是 Apache-2.0 的原模型。
- 接口：ONNX 图的接口固定为 XLS-R → 调节器 → DiT → HiFT，码本与统计量在图内，Rust 运行时按目录的 manifest 加载。
- 质量：说话人相似度高、源音色泄漏少；流式可懂度与离线相当；对叠加噪声较敏感（§5）。

## 5. 已知限制

1. **码本模型的调节器统计量必须冻结**。图 2.0.0 用每段自己的 GroupNorm 统计量，流式 WER 16 %；2.0.1 冻结后 7.6 %。以后任何码本模型导出都要带 `--norm-stats`，并先用 `tools/stream_diag.py --segmented` 复现流式 WER 再导。
2. **说话时混入敲击声会吃字**。键盘声在语音之下 10 dB 时，2.0.1 的 WER 从 12 % 升到 31 %。门控管不了说话中的噪声；修法是噪声增强微调，未做。
3. **门只分语音 / 非语音**。别人的说话声、部分咳嗽喷嚏会放行；说话人相似度门未做。
4. 风格编码器是中文 CAM++；英文目标音色用它选参考仍有效，但 2.x 计划换英文模型。
5. 输出 22.05 kHz（上限 11 kHz），由 HiFT 决定。
6. 提示（参考音色）取参考音频前 3 s；CAM++ 图动态形状、不走 CUDA Graph，换声音约 350 ms。
7. 评测只做了英文（LibriTTS-R）；法语、带口音、嘈杂场景未量化。

## 6. 复现与校验

```bash
# 导出（MimickingTrain，Mimicking 的 .venv；整套自给自足；--shared from-weights 为默认）
python -m mimictrain.export --weights runs/Mimicking-v2.0/ema.safetensors --train-config runs/Mimicking-v2.0/config.json \
    --norm-stats runs/Mimicking-v2.0/regulator_norm_stats.json --out ../Mimicking/models/Mimicking-v2.0-lowlat \
    --block 0.25 --right 0.02 --gpu 0 --version 2.0.1
# 对拍 ORT 对 PyTorch
PYTHONPATH=. python tools/export_parity.py --dir ../Mimicking/models/Mimicking-v2.0-lowlat --weights runs/Mimicking-v2.0/ema.safetensors --train-config runs/Mimicking-v2.0/config.json --gpu 0
# 文件校验（Mimicking）
mimic models verify --models models/Mimicking-v2.0-lowlat
# Rust 对拍（夹具在套内 fixtures/；CUDA / cuDNN DLL 目录需在 PATH 上）
MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=1 MIMIC_MODELS=models/Mimicking-v2.0-lowlat cargo test --release -p mimic-model -p mimic-dsp -- --test-threads=1
# 流式口径复现（PyTorch 分段模拟与 Lab 一致：2.0.1 7.2 %）
PYTHONPATH=. python tools/stream_diag.py --segmented --fixed-stats runs/Mimicking-v2.0/regulator_norm_stats.json --weights runs/Mimicking-v2.0/ema.safetensors --config runs/Mimicking-v2.0/config.json --gpu 0 --out runs/check --pairs 32
```

数据与记录：MimickingTrain `docs/despeaker-design.md` §12–13（微调、冻结、Lab 实测、块长、右前瞻、咬词诊断、杂音），`runs/lab_32pair*/`、`runs/diag_*`、`runs/noisectx_*`。
