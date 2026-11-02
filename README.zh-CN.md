[English](README.md) | 中文

# Mimicking

Rust 实现的零样本实时变声库：给它 16 kHz 的麦克风块，它还你换好音色的 22.05 kHz 音频块，只需一段 3 秒的参考录音，不针对目标说话人训练。
链路为 XLS-R 内容编码 → 内容码本 + 长度调节 → DiT 流匹配 → HiFT 声码器，生成器（调节器 + DiT）是 MimickingTrain 自训的 2.0 权重；模型以静态 ONNX 图在 ONNX Runtime 的 CUDA 执行器上以 CUDA Graph 运行，外围全部 Rust。
一张 RTX A4000 上每块 240 ms 音频耗时约 30 ms，单卡 4 路；160 ms 块端到端约 330 ms。

*Mimicking is a Rust, library-first real-time zero-shot voice conversion library: XLS-R content encoding, a content
codebook and length regulator, a DiT flow-matching generator with our own 2.0 weights, and the HiFT vocoder, all as
static ONNX graphs on ONNX Runtime CUDA with CUDA Graph. One RTX A4000 converts a 240 ms block in about 30 ms and
sustains 4 concurrent calls; end-to-end latency is about 330 ms at 160 ms blocks.*

## 先看这些

| | 文档 | 内容 |
|---|---|---|
| 导览 | [项目导览](https://infohex.github.io/Mimicking/overview.zh-CN.html)（源文件 `docs/overview.zh-CN.html`） | 给决策者的图文总览：做什么、怎么做、延迟与容量、里程碑 |
| 流程 | [实时链路流程图](https://infohex.github.io/Mimicking/pipeline-flow.zh-CN.html) | 一块 240 ms 音频怎样变成输出：各阶段的输入输出、耗时、代码位置、术语表 |
| 目录 | [目录结构](https://infohex.github.io/Mimicking/project-tree.zh-CN.html) | 三个工作区每个目录与文件的说明，可搜索 |
| 发布 | [`docs/release-2.0.zh-CN.md`](docs/release-2.0.zh-CN.md) | 2.0 冻结清单、最终参数、评测数字、已知限制 |
| 接入 | [`docs/integration-guide.zh-CN.md`](docs/integration-guide.zh-CN.md) | 其他项目如何接入 |
| 运行时 | [`docs/runtime-pitfalls.zh-CN.md`](docs/runtime-pitfalls.zh-CN.md) | ORT / CUDA / 多线程的硬规则 |
| 许可 | [`docs/license-review.zh-CN.md`](docs/license-review.zh-CN.md) | 逐项许可审读；库代码 Apache-2.0 |

三个交互页在 GitHub 上直接点开只显示源码，请用上面的在线地址（GitHub Pages），或克隆仓库后用浏览器打开 `docs/*.html`。

**2.0 正式版（2026-11-01）**：默认模型套 `models/Mimicking-v2.0-lowlat`（240 ms 块；另有 160 / 320 ms）；语音门为 Silero VAD；32 对评测上英文说话人模型
像目标 0.712 / 像源 0.419，流式 WER 7.6 %。
冻结清单、最终参数与数字见 **`docs/release-2.0.md`**；接入方法见 **`docs/integration-guide.md`**；训练与调试踩过的坑见 MimickingTrain `docs/lessons-learned.md`；
三个工作区的全部目录与文件说明见交互页 **`docs/project-tree.html`**（`python tools/project_tree.py` 重新生成）。

## 结构

```
mimic-cli        命令行：convert（文件到文件）、stream（多路实时模拟）、models（清单校验）
mimic-pipeline   EulerCfg 求解器、Session/Runner 流式会话、Converter；只依赖 trait
mimic-core       阶段 trait：Graph、ContentEncoder、SpeakerEncoder、LengthRegulator、VelocityEstimator、FlowStepper、Vocoder、MelAnalyzer、Resampler、Vad
mimic-dsp        重采样（torchaudio 算法）、梅尔谱、Kaldi fbank、HiFT 谐波源/STFT/ISTFT、SOLA、能量 VAD
mimic-onnx       ONNX Runtime 后端：运行时加载 dylib、设备驻留绑定、CUDA Graph
mimic-model     模型包（XLS-R → 调节器 → DiT → HiFT 这一族）：各 trait 的实现、窗口几何、manifest 校验、into_stages()
examples/        convert-minimal：50 行以内的完整转换程序
tools/           package_runtime.py（装配运行目录）、project_tree.py（目录结构页）
```

依赖方向自上而下：`mimic-core` 和 `mimic-pipeline` 不依赖任何模型或运行时；`mimic-model` 是第一个模型包，别的模型族可以并列。

## 快速开始

```rust
use mimic_pipeline::{Converter, EulerCfg, Runner, Session};
use mimic_model::{Device, ModelFiles, OnnxRuntime, ModelSet};

let rt = OnnxRuntime::discover(None, Device::Cuda(0))?.with_cuda_graph(true);
let files = ModelFiles::new("models/Mimicking-v2.0-lowlat");          // 2.0 默认套
let model = ModelSet::load(&rt, &files, files.config())?;              // 几何从目录的 manifest 读
let (stages, geo) = model.into_stages();
let mut runner = Runner::new(stages);
let prompt = runner.stages.prepare_prompt(&reference_22k, &geo)?;          // once per voice
let mut session = Session::new(geo, prompt, EulerCfg { steps: 10, cfg: 0.7 }, 1234);
let out_22k = Converter::convert(&mut runner, &mut session, &source_16k, |_, _| {})?;
// streaming: runner.push(&mut session, &block_16k)? -> Vec<f32> of block_frame samples at 22.05 kHz
```

完整程序见 `examples/convert-minimal/src/main.rs`。命令行：

```bash
mimic convert --reference ref.wav --source src.wav --out out.wav
```

```bash
mimic stream --reference ref.wav --source a.wav b.wav c.wav --realtime --vad --out-dir out/
```

```bash
mimic models verify
```

## 模型与运行库

`models/` 下每个子目录一套静态图 + `manifest.json`（manifest `name` 为 `Mimicking-v2.0`，码本与冻结的调节器统计量已烘入图内，见 `docs/release-2.0.md`）；运行时按目录的 manifest 加载。Silero 语音门在 `models/vad/`。

库不打包模型。套由 MimickingTrain 的 `python -m mimictrain.export` 整套导出：调节器与 DiT 来自我们的权重，XLS-R、CAM++、HiFT
三张图从 `ref/models/` 里的 Apache-2.0 权重导出（XLS-R 按 Hugging Face 名字 `facebook/wav2vec2-xls-r-300m` 加载），对拍夹具写在套内 `fixtures/`。
换块长 / 右前瞻 = 用 MimickingTrain 重新导一套到新目录。

运行库目录约定（`tools/package_runtime.py` 可从 `.venv` 自动装配）：

```
<app>/
  mimic.exe
  runtime/onnxruntime/   onnxruntime.dll、onnxruntime_providers_shared.dll、onnxruntime_providers_cuda.dll
  runtime/cuda/          cudart、cublas、cublasLt、cudnn*、cufft（CUDA 12 / cuDNN 9）
  models/Mimicking-v2.0-lowlat/   图 + manifest.json（每套一个目录）
  models/vad/            silero_vad.onnx（Silero VAD，MIT，`mimic_onnx::SileroVad` 的语音门；没有它只能用能量门）
```

多线程、多卡、重载时的规则（一个进程一个 ORT 环境、CUDA Graph 录制不可与其他 CUDA 调用并发、一卡一线程等）见 `docs/runtime-pitfalls.md`，改动相关代码前先读。

查找顺序：`--ort` 显式路径 → `$MIMIC_ORT_DIR` → `<exe>/runtime/onnxruntime` → `$ORT_DYLIB_PATH`；`<exe>/runtime/cuda` 和 `$MIMIC_CUDA_DIR` 会被加到 `PATH` 前面。`ort` crate 以 `load-dynamic` + `api-22` 对接 onnxruntime 1.22。

## 测试

```bash
cargo test --workspace --release
```

纯 Rust 对拍（重采样、梅尔谱、fbank、HiFT STFT/ISTFT）读取 `MIMIC_FIXTURES`，否则 `<MIMIC_MODELS 或 models/Mimicking-v2.0-lowlat>/fixtures`（mimictrain.export 写在套内）；没有夹具时跳过。ONNX 侧对拍需要 GPU：

```bash
MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=0 cargo test -p mimic-model --release -- --nocapture --test-threads=1
```

## 数字（A4000，10 步，CFG 0.7）

2.0 正式版（`Mimicking-v2.0-lowlat`，CUDA Graph，单路；MimickingLab 实测）：每块 29–30 ms，与块长无关（160 / 240 / 320 ms 套 GPU 占用约 19 / 12 / 9 %）；
32 对 LibriTTS-R 流式 WER 7.6 %，英文 CAM++ 像目标 0.712 / 像源 0.419；160 ms 块、GPU 锁频、50 ms 输出余量时
软件链路延迟 234 ms，加算法固有 80 ms 与声卡缓冲，端到端约 330–350 ms。完整表见 `docs/release-2.0.md`。

每路成本不随路数变化（DiT 在此序列长度上是算力瓶颈），同一张卡上各路顺序执行，每路的处理延迟等于整轮耗时，所以**每张 A4000 最多 4 路**（4 路一轮 115 ms；8 路一轮 230 ms 贴着周期，延迟翻倍且无余量，不可用）；两卡 8 路每卡一轮约 120 ms。完整的并发容量样板（RTX A4000 16 GB，与 RTX 3070 同级；单卡 1/2/4/6/8 路、双卡 8/16 路、功耗、显存、换卡估算方法）见 `docs/capacity-a4000.md`。逐级对拍全部在 fp16 噪声内。复刻阶段的记录（M0–M3、导出侧优化、最初的计划）在 `docs/history/`。

## 许可

代码：Apache-2.0（`LICENSE`、`NOTICE`）。本仓库只包含自写代码，不分发任何模型权重或导出的图。另行获取的模型按各自许可：2.0 的调节器与 DiT 权重为 MimickingTrain 自训；XLS-R、HiFT、CAM++ 为 Apache-2.0；Silero VAD 为 MIT。逐项审读见 `docs/license-review.md`。
