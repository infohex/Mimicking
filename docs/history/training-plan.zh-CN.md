[English](training-plan.md) | 中文

# 自训权重：训练代码分析与自研方案（提案，2026-10-27）

目标：用我们自己的代码从零训练 DiT 流匹配估计器和长度调节器，产出权重归我们所有；训练流水线本身不依赖
seed-vc 或其它 GPL / 非商用代码；比 seed-vc 原版训练脚本快一个数量级。本文先分析原版训练路径逐个文件的来源与问题，
再给出方案，**未经确认不写代码**。

## 一、seed-vc 原版训练路径的构成与来源

训练只更新两个模块：`cfm`（DiT 估计器）和 `length_regulator`。XLS-R、CAM++、HiFT、OpenVoice 冻结。

| 文件 | 行数 | 内容 | 来源与许可 | 处置 |
|---|---|---|---|---|
| `train.py` | 437 | 训练循环、特征提取、音色扰动、损失 | seed-vc 原创，GPL-3.0 | 不用，重写 |
| `data/ft_dataset.py` | 126 | 目录扫描、librosa 加载、梅尔谱、整理批次 | seed-vc，GPL-3.0 | 不用，重写 |
| `optimizers.py` | 119 | 多优化器包装、AdamW、ExponentialLR | seed-vc，GPL-3.0 | 不用，重写（几十行） |
| `modules/commons.py` | 476 | 工具函数、`build_model`、`load_checkpoint` | 大部分来自 VITS（MIT） | 不用 |
| `modules/diffusion_transformer.py` | 536 | DiT：Transformer/Attention/FFN/RMSNorm/RoPE 来自 Meta gpt-fast（文件头标 Meta 版权，BSD-3）；`TimestepEmbedder` 注释指向 OpenAI GLIDE（MIT）；`modulate`、`FinalLayer`（adaLN）与 facebookresearch/DiT 的 `models.py` 逐字相同，**该仓库许可为 CC BY-NC 4.0（非商用）**，需核实 | 混合来源，含疑似非商用片段 | 不用，按架构重写 |
| `modules/flow_matching.py` | 167 | CFM 损失与欧拉采样 | 结构与注释来自 Matcha-TTS / CosyVoice（MIT / Apache-2.0） | 不用，自写（约 60 行） |
| `modules/length_regulator.py` | 141 | 线性投影 + 最近邻插值 + Conv/GroupNorm/Mish ×4 + 1×1 卷积 | CosyVoice（Apache-2.0）；import 了 descript-audio-codec 的 VQ（MIT，tiny 配置不用） | 不用，自写（约 30 行） |
| `modules/audio.py` | 82 | 梅尔谱 | HiFi-GAN（MIT） | 不用，已有 Rust 版 `MelSpec` 的定义，Python 端对照它重写 |
| `modules/openvoice/*` | ~1000 | 音色扰动增强用的 VITS 转换器 | OpenVoice V2（MIT），权重 MIT | 不用，换自写扰动 |
| `modules/campplus/*` | ~400 | CAM++ 说话人编码器 | 3D-Speaker（Apache-2.0） | 不用，用我们导出的 `campplus.onnx` |
| `modules/hifigan/*` | ~500 | HiFT 声码器 | CosyVoice（Apache-2.0） | 训练不需要；评估用我们导出的 ONNX |
| transformers `Wav2Vec2Model` | — | XLS-R 内容编码器 | Apache-2.0 | 保留（唯一保留的第三方模型代码） |

结论：训练路径上值得保留的只有 XLS-R（通过 transformers）和三份 Apache-2.0 权重。其余全部可以按架构重写，而且有
必要重写：seed-vc 的 DiT 文件本身就是 gpt-fast + GLIDE + facebookresearch/DiT 的拼接，后者的 CC BY-NC 许可
与 GPL 仓库许可矛盾，拿来用等于继承了一个说不清的许可链。

## 二、seed-vc 原版训练脚本的问题（为什么不该直接用）

按 `train_one_step` 逐步看（batch 2，每批最长 100 s 音频）：

1. **每一步都重算冻结模型**：OpenVoice 做一遍音色转换（含 STFT、VITS 推理），XLS-R 跑两遍（原始 + 扰动，300M 参数，fp16），
   CAM++ 在 Python 循环里逐条算 fbank 和嵌入。这些占了一步 90% 以上的时间，而 DiT 本身只有 25M 参数。
2. **数据加载在主进程**：librosa 逐文件解码（Windows 下 `num_workers=0`），每条重算梅尔谱，无长度分桶，批内按最长补零。
3. **DiT 全程 fp32**，没有 autocast / bf16，没有 torch.compile；注意力用 SDPA 但 mask 被扩成 `(B,1,T,T)` 的大张量。
4. **CFG 丢弃按整批**（`torch.rand(1)`），不是逐样本；时间步均匀采样。
5. **没有 EMA**：发布的检查点文件名是 `_ema`，说明作者自己的训练有 EMA，但公开的 `train.py` 没有；照用这个脚本
   连作者的配方都复现不了。
6. 没有验证集、没有评估、没有预热、学习率只有 ExponentialLR（每步 ×0.999996），没有梯度累积，v1 不支持多卡。
7. 损失和 prompt 掩码用 Python 按样本循环。
8. 依赖链臃肿：munch、librosa、descript-audio-codec、modelscope、funasr、gradio 等，训练并不需要。

这些问题叠加，单卡 A4000 上估计每步 0.5–1 s，一半以上花在冻结模型上。

## 三、方案

### 3.1 代码落点与依赖

独立 Python 项目 `C:\Workspace\Rust\MimickingTrain`（与 MimickingLab 同样的理由：不混进库仓库；模型定义同时作为
新的导出路径被 Mimicking 的 `tools/export` 复用）。依赖只留：torch、torchaudio（BSD）、transformers（Apache，只为
XLS-R）、safetensors（Apache）、numpy、soundfile（BSD）、onnxruntime-gpu（MIT，评估时跑 CAM++ 与 HiFT 的 ONNX）。
不引入 seed-vc、OpenVoice、descript-audio-codec、librosa、munch、modelscope、funasr。

### 3.2 模型定义（自写，约 400 行）

按架构规格实现，不翻译 seed-vc 原版文件：

- **DiT**：384 维、9 层、6 头；RMSNorm + 时间嵌入调制（adaLN 形式）；RoPE；SwiGLU 前馈（中间维按 gpt-fast 的
  2/3·4d 取整到 256 倍数，保持与基线权重形状一致）；U-ViT 跳连（前 4 层输出接到后 4 层，线性融合）；时间和音色
  各占一个 token；输入为 [x, prompt_x, cond(, style)] 拼接后线性投影；输出 MLP。
- **长度调节器**：Linear(1024→384) → 最近邻插值到声学帧率 → (Conv3 + GroupNorm + Mish)×4 → Conv1。
- **CFM**：线性路径 y = (1−(1−σ)t)·z + t·x₁，目标 u = x₁ − (1−σ)z，L1 损失，prompt 区域置零、不计损失；
  欧拉采样与 CFG 与 Rust 端完全一致。

**验收**：写一个 key 映射把基线 tiny 权重装进我们的定义，与基线前向逐层对拍（相对 L2 在 fp32 下 ≤ 1e-5）。
通过即证明架构等价，之后导出的 ONNX 仍满足 Rust 端现有的图接口（x, prompt_x, t, style, mu → v；融合版同理），
Rust 一行不改。对拍只在开发机上做，基线权重不进训练。

### 3.3 数据流水线

- 清单构建：扫描目录，解码为 22.05 kHz 单声道，时长 1–30 s，响度归一（−23 LUFS），可选 VAD 切段；写 parquet/jsonl 清单。
- **离线缓存小特征**：梅尔谱（与 Rust `MelSpec` 同定义：1024/256/80、slaney、log(max(x,1e-5)））、CAM++ 嵌入
  （我们的 `campplus.onnx`）。每小时音频约 60 MB，1000 h 约 60 GB。
- **XLS-R 在线算**（bf16、no_grad、按长度分桶后成批、torch.compile）：1024 维 50 Hz 特征每小时 360 MB，缓存
  1000 h × (原始 + 扰动) 要 700 GB 以上，不划算；在线算一批 160 s 音频约 0.2–0.3 s，可以接受。可选 int8 缓存给
  反复实验的子集。
- **音色扰动自写**：目的只是让内容特征少带音色。实现 NANSY 式扰动：共振峰移位（重采样 + 时长还原）、基频移位、
  随机参数 EQ、随机增益，全部在 GPU 上用 torch 做，每步换随机参数。不用 OpenVoice，也就不需要 `se_db.pt`。
  prompt 区域用原始特征、其余用扰动特征，与基线一致。

### 3.4 训练器

- PyTorch 2.x，bf16 autocast，torch.compile，SDPA（flash），逐样本 CFG 丢弃（10%），时间步可选 logit-normal 采样。
- 按长度分桶的动态批（按帧预算，如每批 20k 帧 ≈ 4 分钟音频），梯度累积，AdamW（β 0.9/0.98，wd 0.01）+ 线性预热 +
  余弦衰减，梯度裁剪 1.0，**EMA**（0.9999）。
- DDP：`torchrun` 跑两张 A4000（各自 16 GB 够用）。
- 检查点与断点续训（safetensors + 优化器状态），验证集损失，每 N 步用 HiFT ONNX 合成固定测试句，算 CAM++ 余弦，
  TensorBoard 记录。
- 预计：缓存就绪后单卡每步约 0.3–0.4 s（其中 XLS-R 在线约一半），双卡 DDP 翻倍吞吐；30 万步约 1–2 天。

### 3.5 导出与评估

- 导出：把 `export_dit_opt.py` 里融合算子的 symbolic 写法移到我们的模型定义上，输出同名的 ONNX 文件与 manifest；
  导出脚本从此不再 import seed-vc。
- 评估：固定测试集（与现有 `eval_outputs.py` 同口径的 CAM++ 余弦、Whisper WER）对比基线 tiny；测试台跑实时与 4 路负载。

### 3.6 分期

| 期 | 内容 | 产出 | 预计 |
|---|---|---|---|
| T0 | 模型定义 + 基线权重对拍 + 新导出路径 | 架构等价证明；tools/export 去 GPL 依赖 | 2–3 天 |
| T1 | 数据流水线 + 扰动 + 特征缓存 | 1 个可商用数据集的清单与缓存 | 3–5 天（含下载） |
| T2 | 训练器 + 单卡冒烟（小数据过拟合） | 训练能收敛的证据 | 2–3 天 |
| T3 | 全量训练（双卡） + 评估 + 导出 | 第一版自有权重 + 评估报告 | 训练 1–2 天，评估 1–2 天 |

数据前提：可商用语音集。英文：LibriTTS-R（CC BY 4.0）、MLS（CC BY 4.0）；中文：AISHELL-3（Apache-2.0）太小，
需另行授权或自采。这一项在 T1 前要定。

## 五、进度

- **T0 完成（2026-10-27）**：`C:\Workspace\Rust\MimickingTrain`。自写 DiT 与长度调节器对基线模块逐位/1e-7 级对拍通过；
  导出路径 `python -m mimictrain.export` 不 import seed-vc，产出的图在 ORT 和 Rust 端（`mimic convert`）验证通过，
  与旧导出音色余弦一致。发现：seed-vc 原版 cfm 的 25.68 M 参数里 7.4 M 是前向不用的死模块，有效参数 18.27 M。
  BO 已决定：代码放 MimickingTrain，第一版英文公开数据。下一步 T1（数据流水线）。

- **T1 完成（2026-10-27）**：下载器（断点续传、md5、EU 镜像 33 MB/s）、清单（1–30 s 过滤）、与 Rust/基线同定义的梅尔谱
  （对基线 < 1e-5）、CAM++ fbank、safetensors 分片缓存（int16 波形 + fp16 梅尔谱 + fp16 音色向量）、自写音色扰动
  （基频 ±1.5–6 半音、共振峰 ±8–25%、随机 EQ、增益；原始与扰动的 CAM++ 余弦中位约 0.35）。
  LibriTTS-R 七个子集全部缓存并逐条核对（缓存 119 GB，`C:\MimickingData`；压缩包曾在 USB 机械盘，缓存被寻道拖住后整体
  搬到 SSD，详见 MimickingTrain README）。下一步 T2（训练器）。

| 子集 | 条数 | 小时 | 说话人 |
|---|---|---|---|
| train-clean-100 | 31717 | 53.1 | 247 |
| train-clean-360 | 110924 | 188.9 | 899 |
| train-other-500 | 194851 | 306.0 | 1160 |
| **训练合计** | **337492** | **548.0** | **2306** |
| dev-clean / dev-other | 5490 / 4353 | 8.9 / 6.3 | 40 / 33 |
| test-clean / test-other | 4615 / 4679 | 8.5 / 6.6 | 38 / 33 |
| 全部 | 356629 | 578.3 | — |

- **T2 完成（2026-10-27）**：训练器 `mimictrain/train/`（bf16、按长度分桶的帧预算批、逐样本 CFG 丢弃、EMA、两卡 DDP（gloo）、
  在线 XLS-R 原始 + 扰动两遍、检查点与续训、验证损失、HiFT 合成 + CAM++ 相似度评估）。长度调节器改为按样本长度插值、
  掩码 GroupNorm，成批训练与单句推理逐位一致。扰动改成全成批实现（2.5 s → 37 ms/批）。两卡实测约 400× 实时，
  一个 epoch（548 h）约 1.3 h，默认 10 万步约 1.5 天。基线：原版权重验证损失 0.606、CAM++ out↔ref 0.848。下一步 T3（正式训练）。

### 算力实测与训练时长估计（2026-10-27 上午的粗测，单张 A4000；T2 完成后的准确数字见上一条与 MimickingTrain README）

| 环节 | 实测 | 折合 |
|---|---|---|
| DiT + 长度调节器训练步（bf16，未 compile） | 16 × 10 s = 160 s 音频 / 536 ms | 约 300× 实时 |
| XLS-R 12 层前向（bf16，冻结），每步要跑两遍（原始 + 扰动） | 16 × 10 s / 280 ms | 约 580× 实时，两遍后约 290× |
| 合计每卡 | | 约 140× 实时 |
| 两卡 DDP | | 约 270× 实时 |

LibriTTS-R 训练子集 548 h：一个 epoch 约 2.0 h。按 20 epoch（够出第一版可听结果）约 2 天，按 40–50 epoch（约 30 万步，
每步 320 s 音频）约 4–5 天。seed-vc 原版脚本单卡约 40–80× 实时，无多卡；我们约快 3–7 倍。进一步提速的手段：torch.compile、
只对原始音频缓存 XLS-R 特征（省掉一遍前向）、更大 batch，合计有望再快 1.5–2 倍。

## 四、需要确认的决定

1. 方案整体是否认可；代码放独立目录 `MimickingTrain` 是否同意。
2. 第一版先做英文（数据现成）还是直接中文（先解决数据）。
3. T0 的对拍允许在开发机上加载基线权重做验证（只验证、不训练、不分发）。
