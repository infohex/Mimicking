[English](rust-realtime-vc-plan.md) | 中文

# Mimicking：独立复刻并优化 seed-vc 的 Rust 实现方案（v3）

状态：提案 v3，未动工。
- v1：独立桌面工具。v2：直接做成 RPIVoiceGW 的 worker 替代品。
- **v3（BO 2026-10-26 定）**：Mimicking 是独立项目，先复刻并优化 seed-vc，模块化设计自己定边界，不逐行照抄；移植成功后再以模块化方式被其他项目复用，RPIVoiceGW 是第一个候选使用方，但不在本阶段范围内。

依据：`bench/RESULTS.md` 实测。

## 0. 一句话

一个 Rust workspace，核心是一组**以 trait 定义边界的库 crate**：内容编码、音色编码、长度调节、流匹配去噪、声码器、重采样、VAD 各是一个可替换模块，由一个管线对象组合成"参考音频 + 源音频 → 转换后音频"的离线接口和"推块进、推块出"的流式接口。第一套实现对应 seed-vc tiny 的权重，模型以 ONNX 静态图在 ONNX Runtime CUDA 执行器加 CUDA Graph 上运行。可执行文件只是薄壳：本阶段只做一个用于验证和测量的命令行工具；给网关或别的项目用的适配层等移植成功后再定。

## 1. 原则

| 原则 | 含义 |
|---|---|
| 库优先 | 一切能力在 crate 里，二进制只做参数解析和 I/O。任何使用方（命令行、网关 worker、以后别的服务）都是薄适配层 |
| 边界自己定 | 按算法阶段切模块，不按 seed-vc 原版 Python 文件切。原版的 `real-time-gui.py` 把窗口管理、模型调用、SOLA 揉在一个回调里，这里拆开 |
| 权重兼容，代码不抄 | 必须能加载基线权重（否则没法对拍、没法零样本）；但求解器、缓存、流式状态机、合批是自己的设计 |
| 后端可换 | 推理后端是 trait，第一套用 ORT；以后 TensorRT、DirectML，甚至别的模型族都不改上层 |
| 一切可测 | 每个模块有对拍夹具；管线有端到端指标；流式有每段计时 |
| 不预设使用方 | 核心 crate 不知道什么是 8 kHz 电话、什么是管道协议、什么是 config.ini |

## 2. 实测给出的约束（不变）

| 实测 | 要求 |
|---|---|
| DiT 单次前向 14.2 ms，CPU 下发 2619 算子 13.95 ms，CUDA Graph 回放 3.6 ms | 模型整图执行，不逐算子调用 |
| 序列 128 到 1024 帧耗时不变；batch 1 和 2 相同（PyTorch eager） | 固定形状。**M0 修正**：这是调度开销盖住真实计算的假象，图执行后 DiT 耗时随 batch 线性增长（每路每步约 5 ms），合批没有吞吐收益，见 `m0-results.md` §3 |
| `sequence_mask()` 的 `.max()` 每步同步 | 导出时去掉 |
| XLSR、HiFT 同受调度拖累 | 全部走静态图 |
| 纯 CPU 外围 7 ms | 外围不是重点，但要和 GPU 提交分线程 |

## 3. 架构

### 3.1 crate 与依赖方向

```
mimic-cli（二进制，验证与测量用）
   │
mimic-pipeline      离线转换 + 流式会话 + 合批；只依赖下面的 trait
   │
mimic-core          trait 与数据类型：ContentEncoder、SpeakerEncoder、LengthRegulator、
   │                FlowDenoiser、Vocoder、Resampler、Vad、Backend；Audio、Mel、Features 等类型
   ├── mimic-dsp         纯 Rust 信号处理：重采样、梅尔谱、Kaldi fbank、ISTFT、SOLA、淡入淡出、能量 VAD
   ├── mimic-onnx        Backend 的 ORT 实现：会话、固定形状缓冲、CUDA Graph、设备选择、计时
   └── mimic-model      seed-vc tiny 的"模型包"：各 trait 的实现（组合 ORT 图 + dsp）、形状计算、
                         配置（块长、上下文、提示长度、步数、CFG）、权重清单与校验
tools/export/（Python，用 .venv）  从 ref/seed-vc 导出 ONNX、生成对拍夹具
```

规则：`mimic-core` 和 `mimic-pipeline` 不依赖 ORT、不依赖任何模型；`mimic-model` 是第一个模型包，以后别的模型族是并列的 crate。

### 3.2 核心 trait（意图，不是最终签名）

| trait | 输入 → 输出 | seed-vc tiny 实现 |
|---|---|---|
| `ContentEncoder` | 16 kHz 波形窗口 → 50 Hz 内容特征 | XLSR-300m 前 12 层（ORT） |
| `SpeakerEncoder` | 16 kHz 参考波形 → 音色向量 | Kaldi fbank（dsp）+ CAM++（ORT） |
| `LengthRegulator` | 内容特征 + 目标帧数 → 条件序列 | 最近邻插值 + 卷积（ORT） |
| `FlowDenoiser` | 条件 + 提示梅尔 + 音色 + 噪声 + 步数/CFG → 梅尔 | 单步 DiT 图（ORT）+ Rust 求解器（欧拉；可换） |
| `Vocoder` | 梅尔 → 波形 | HiFT 主体（ORT）+ ISTFT（dsp） |
| `Resampler` / `Vad` | 波形 → 波形 / 门控 | dsp |
| `Backend` | 加载图、绑定固定形状缓冲、运行、是否启用图捕获 | ORT CUDA EP |

`mimic-pipeline` 之上的两个对象：
- `Converter`：离线。`prepare_reference(wav) -> Prompt`（可缓存、可序列化）；`convert(prompt, wav) -> wav`。
- `StreamSession`：流式。按块长、上下文、提示长度构造，`push(block_16k) -> Option<block_out>`；内部管理滚动窗口、SOLA、VAD 门控、每段计时。`BatchRunner` 把多条 `StreamSession` 的同一周期合到一次图运行。

输出采样率是模型的 22.05 kHz；降到 8 kHz、切 20 ms 帧之类的事属于使用方，`mimic-dsp` 提供重采样器但管线不替使用方做决定。

### 3.3 与基线不同的设计点

- 单步去噪图 + Rust 求解器：步数、CFG、求解器都在 Rust 侧可变，图不变。
- `Prompt` 是显式对象而不是全局变量；可按 key 缓存，也可以存盘复用。
- 流式状态机独立于音频 I/O 和进程边界，纯函数式接口，便于测试和合批。
- 合批保留为调度手段（一次图运行服务多路，少几次提交），不是吞吐手段：M0 实测 DiT 每路每步约 5 ms 且随 batch 线性增长。固定 `max_batch`，不满补零。
- 掩码、位置编码、`freqs_cis` 在导出时按形状烘成常量，运行时没有同步点。
- VAD 是 trait，默认给能量门限实现；使用方可换。

### 3.4 模型导出（`tools/export/`）

| 模块 | 导出方式 | 固定形状（以块 0.25 s 为例） |
|---|---|---|
| XLSR 前 12 层 fp16 | HF 导出后截层 | N×76800，输出 N×239×1024 |
| 长度调节器 | 导出，目标长度常量 | N×T_cond×384 |
| DiT 单步 | 只导单步；CFG 的 batch 2 进图；掩码常量 | 2N×80×(258 提示帧 + T_cond + 2) |
| CAM++ | 导出，动态长度 | 1×T×80 |
| HiFT | 主体到幅度+相位谱；ISTFT 在 Rust；谐波源导不出则也搬到 Rust | N×80×T_mel |

导出脚本只读 `ref/seed-vc` 和其 checkpoints，必要的改动（去 `.max()`、常量掩码）写在导出脚本自己的模型拷贝里，不改 `ref/`。

### 3.5 运行时选型

| 方案 | 结论 |
|---|---|
| **ONNX Runtime（`ort`）CUDA EP + CUDA Graph** | 首选：图执行解决实测瓶颈；fp16；TensorRT EP 与 DirectML EP 同一套代码 |
| candle | 无图捕获，Rust 逐算子下发 3 到 5 ms 与 GPU 时间同量级；放弃 |
| burn | Windows CUDA 后端太新；观望，`Backend` trait 留口 |
| tch-rs | 同一个分发器换语言调用，不解决问题；放弃 |

### 3.6 配置

一个类型化的 `WindowConfig`（块长、CE 上下文、DiT 上下文、右上下文、提示长度、步数、CFG、最大 batch、设备）。块长、上下文、提示长度、最大 batch 决定图形状，是构造期参数；步数、CFG 运行时可改。默认值与 seed-vc 原版 GUI 一致（0.25 s / 2.5 / 0.5 / 2.0 / 3 s / 10 / 0.7）以便对拍。

## 4. 验证

### 4.1 逐级对拍（`tools/export/` 产夹具，Rust 测试消费）
重采样、梅尔谱、fbank 相对误差 ≤ 1e-3；各 ONNX 图对 PyTorch fp16 最大绝对误差 ≤ 1e-2；整条链对同一噪声种子，输出梅尔谱 L1 ≤ 阈值。

### 4.2 端到端指标
对固定的一组源句和参考，Rust 版与 Python 版输出用同一套客观指标比较：说话人相似度（Resemblyzer 或 CAM++ 余弦）、WER（Whisper 转写）。要求差异在噪声范围内。

### 4.3 性能验收门（A4000，块 0.25 s，10 步，CFG 0.7，单路）

| 指标 | 现状（Python） | 目标 | M0 在 ORT 里的实测（GPU 图合计） |
|---|---|---|---|
| 单路每块耗时 p95，10 步 | 258 ms（带 VAD） | ≤ 80 ms | 70.7 ms + 主机 DSP |
| 单路每块耗时，6 步 | 约 170 ms | ≤ 60 ms | 50.9 ms |
| DiT 10 步 | 155 ms | ≤ 55 ms | 52 到 56 ms（含主机侧欧拉更新） |
| 多路 | 不支持 | 每路成本不高于单路 × 1.1；路数 = 块长 × 0.6 / 每路整块 | DiT 线性：每路每步 5 ms |
| 进程启动到可推理 | 5 到 12 s | ≤ 5 s | 未测（Python ORT 会话创建约数秒） |

## 5. 里程碑

### M0：导出与运行时验证（Python）— 已完成 2026-10-26，见 `m0-results.md`
导出五个图；Python onnxruntime-gpu 里 CUDA EP + CUDA Graph 跑整块；量单路和 batch 2/4/8/16；对拍 PyTorch。
**结果**：DiT 单步 4.95 ms（门 ≤ 5）；整块 10 步 70.7 ms（门 ≤ 80）；精度全部相对 L2 ≤ 4e-3；"batch 4 ≤ 120 ms"的门因前提错误作废，改为每路成本。
遗留优化项（M1 之后）：fp16 权重直出去 Cast、ORT 注意力融合、TensorRT EP（本机未装运行库）。

### M1：核心库 + 离线转换 — 已完成 2026-10-26，见 `m1-status.md`
`mimic-core`、`mimic-dsp`、`mimic-onnx`、`mimic-model`、`mimic-pipeline`（求解器、流式会话、Converter）、`mimic-cli convert`。
**结果**：dsp 六项与 ORT 六项逐级对拍全部通过；端到端 CAM++ 余弦 0.661（PyTorch 流式版 0.647）；每块 147 ms（eager，未开 CUDA Graph）对 PyTorch 199 ms。

### M2：流式会话与性能 — 已完成 2026-10-26，见 `m2-status.md`
设备驻留 I/O 绑定 + CUDA Graph、主机 DSP 提速、融合欧拉步进、`Runner`/`Session` 多路、能量 VAD、`mimic stream`。
**结果**：单路每块 82 ms p50 / 85 ms p95；实时节拍单路零超时；合批按 M0 结论不做。
**导出侧 GPU 优化（同日，见 `gpu-opt.md`）**：DiT 重写前向发射 ORT 融合算子、整图 fp16，单步 4.95 → 2.10 ms；HiFT 主体 fp16 加防溢出 Snake 14.3 → 10.2 ms。单路 **48 ms p50 / 53 ms p95**（门 80 通过）；A4000 上 10 步 4 路、6 步 5 路。

### M3：作为库发布 — 已完成 2026-10-26，见 `m3-status.md`
README、rustdoc 零警告、`examples/convert-minimal`（40 行）、`manifest.json` + `mimic models verify`、`OnnxRuntime::discover` 运行库目录约定、`tools/package_runtime.py`（3.0 GB 自包含目录，干净环境验证通过）、GitHub Actions CI（Windows + Ubuntu，手动 GPU 作业）、CHANGELOG 0.1.0、首次 git 提交。
**验收门**：通过。待定：代码许可定稿与 LICENSE 文件。

### M4：使用方适配（移植成功后另行决定）
例如 RPIVoiceGW 的 worker 协议适配器：薄壳，调用 `StreamSession` 和 `BatchRunner`，做 22.05 k → 8 k 和协议帧。放哪个仓库、改什么配置，到时候再定。

### M5：模型层加速（可选，需训练或重导出）
步数蒸馏、CFG 蒸馏、TensorRT int8、提示段 KV 缓存。

## 6. 风险

| 风险 | 对策 |
|---|---|
| HiFT 的 ISTFT、谐波源导不出 | 已预案：Rust 实现 |
| ORT 的 CUDA Graph 对某算子静默退回 | M0 第一件事就量，用 ORT profiling 确认 |
| XLSR 在 ORT 的真实 GPU 时间偏高 | M0 同测；备选 TensorRT EP |
| trait 边界过早抽象导致性能损失（跨模块拷贝） | 数据类型用零拷贝切片 + 预分配缓冲；基准测试守住 §4.3 |
| GPL-3.0 与权重许可 | 按论文和权重重写，不含 seed-vc 原版代码；权重许可动工前查 Hugging Face |
| seed-vc 原版已停更 | 导出只依赖 `ref/` 快照和固定权重 |

## 7. 待拍板

1. M0 用 GPU 1（GPU 0 留给别的实验），默认如此。
2. 目录：`crates/` 放上述 crate，`tools/export/` 放 Python，`models/` 放导出的 ONNX（不入库），`bench/`、`docs/`、`ref/` 保持。
3. 是否现在就开始 M0。
