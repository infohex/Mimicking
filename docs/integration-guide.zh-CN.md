[English](integration-guide.md) | 中文

# Mimicking 接入指南

给要在自己的产品里用 Mimicking 做实时变声的项目。Mimicking 是一个 Rust 库：你给它 16 kHz 的麦克风块，它还你 22.05 kHz 的变声块；音频设备、队列、门控策略、配置文件都在使用方手里。本文按接入顺序写，每节末尾是"必须做对的事"。

版本与数字见 `docs/release-2.0.md`；运行时规则的来龙去脉见 `docs/runtime-pitfalls.md`。

## 1. 运行环境

```
<app>/
  your.exe
  runtime/onnxruntime/   onnxruntime.dll、onnxruntime_providers_shared.dll、onnxruntime_providers_cuda.dll（1.22）
  runtime/cuda/          cudart、cublas、cublasLt、cudnn*、cufft（CUDA 12 / cuDNN 9）
  models/Mimicking-v2.0-lowlat/        默认套（240 ms 块）
  models/Mimicking-v2.0-b0.16-r0.02/   160 ms 块（可选）
  models/vad/silero_vad.onnx           语音门
```

查找顺序：`OnnxRuntime::discover(explicit, device)` 依次看显式路径、`$MIMIC_ORT_DIR`、`<exe>/runtime/onnxruntime`、`$ORT_DYLIB_PATH`；`<exe>/runtime/cuda` 与 `$MIMIC_CUDA_DIR` 会加到 `PATH` 前面。`tools/package_runtime.py` 能从 `.venv` 装配这个目录。显存：一套模型约 2.5 GB。

**必须做对**：一个进程只有一个 ONNX Runtime 环境，`OnnxRuntime::load / discover` 第二次调用返回同一个环境，换设备用 `with_device`。CUDA Graph 录制（加载 / 预热）不能与其他线程的 CUDA 调用并发：先把所有卡、所有套加载完，再开始实时推理。

## 2. 选模型套

| 套 | 块长 | 端到端延时（见 §5） | GPU 占用（A4000，单路） | 何时用 |
|---|---|---|---|---|
| `Mimicking-v2.0-b0.16-r0.02` | 160 ms | ≈ 330 ms | 19 % | 延时优先 |
| `Mimicking-v2.0-lowlat` | 240 ms | ≈ 410 ms | 12 % | 默认，容量与延时的平衡 |
| `Mimicking-v2.0-b0.32-r0.02` | 320 ms | ≈ 490 ms | 9 % | 多路并发优先 |

三套同权重、同 0.02 s 右前瞻，音质和可懂度在评测噪声内无差别。同一家族（manifest 的 `name` 相同、右前瞻相同）之间切块长只是换目录重新加载。用 `mimic models verify --models <dir>` 校验文件与 sha256。

## 3. 最小接入代码

```rust
use mimic_pipeline::{EulerCfg, Runner, Session};
use mimic_model::{Device, ModelFiles, OnnxRuntime, ModelSet, SileroVad, CONTENT_SR, SR};

// 一次：运行时 + 模型（几何从 manifest 读）
let rt = OnnxRuntime::discover(None, Device::Cuda(0))?.with_cuda_graph(true);
let files = ModelFiles::new("models/Mimicking-v2.0-lowlat");
let model = ModelSet::load(&rt, &files, files.config())?;
let (stages, geo) = model.into_stages();
let mut runner = Runner::new(stages);

// 每个声音一次：参考音频 22.05 kHz 单声道，取前 3 s
let prompt = runner.stages.prepare_prompt(&reference_22k, &geo)?;

// 每路通话一个会话：门控 + 独立噪声种子
let gate = SileroVad::load(&rt, std::path::Path::new("models/vad/silero_vad.onnx"))?;
let mut session = Session::new(geo, prompt, EulerCfg { steps: 10, cfg: 0.7 }, 1)
    .with_vad(Box::new(gate));

// 实时循环：每次正好 geo.block_16k 个 16 kHz 样本进，geo.block_frame 个 22.05 kHz 样本出
loop {
    let block16: Vec<f32> = capture.next_block(geo.block_16k);      // 你的采集 + 重采样到 16 kHz
    let out22 = runner.push(&mut session, &block16)?;
    playback.push(&resample_22k_to_device(&out22));                 // 你的输出重采样 + 队列
}
```

- `CONTENT_SR` = 16000，`SR` = 22050；`geo.block_16k`（160 ms 块 = 2560，240 ms = 3840）、`geo.block_frame`（22.05 kHz 输出样本数）、`geo.delay_samples()`（算法固有延迟，80 ms）都从几何拿，不要写死。
- 换声音：`session.set_prompt(new_prompt)`，窗口与 SOLA 状态保留，无接缝。
- 离线整文件：`Converter::convert(&mut runner, &mut session, &source16k, |_, _| {})` 走同一条流式路径并对齐去掉延迟；离线不要挂门。
- 完整例子：`examples/convert-minimal/src/main.rs`；命令行 `mimic convert / stream / models`。

**必须做对**：一张卡一个线程一个 `Runner`，多路 = 同一个 Runner 顺序 `push` 多个 `Session`（各路成本不随路数变，一轮耗时 = 各路之和，必须小于块周期）；多卡 = 多个线程各一个 Runner。慢活（CAM++ 嵌入、文件 IO、日志写盘）不要放在推理线程。

## 4. 音频管线

- **输入**：设备采样率 → 16 kHz 单声道，攒满 `block_16k` 再 `push`。采集溢出要丢而不是攒（积压就是延迟）。
- **输出**：22.05 kHz → 设备采样率，进一个环形队列由声卡回调消费。
- **队列控制器**（必须有，否则延迟会随卡顿单向上爬）：每次推块前看队列深度相对目标余量的偏差。
  - 时钟漂移（麦克风与扬声器不同晶振，USB 麦克风常差 0.1–0.3 %）用**播放速率微调**吸收：误差 e（秒）→ 速率修正 c = clamp(e / 4 s + 慢积分, ±1 %)，输出经一级变速重采样（三次插值即可）。人耳听不出 1 %，队列不再漂。
  - 真正卡顿（队列低于余量 1/4 或高于余量 + 两块）才补静音 / 丢输入兜底。每次补或丢都是一次可听的编辑，正常使用应为 0。
  - 不要用"低于余量一半补、高于余量加半块丢"的硬阈值当常规手段：漂移 0.25 % 时每 30 s 剪一刀 80 ms，听感是结巴。
- **GPU 抖动**：单路时 GPU 在块间降频，块耗时 p95 比 p50 高约 60 ms。锁频（`nvidia-smi -i N -lgc 1800,1800`，管理员）后只差 2 ms，余量可从 150 ms 降到 50 ms。多路并发时 GPU 不空闲，天然没有这个问题。
- **麦克风**：关闭系统 / 驱动的"音频增强、自动增益、噪声抑制"，把说话电平调到 −25 dBFS 左右。自动增益会在一阵响的敲击后把增益压低几秒，表现为"敲完键盘几秒说不出话"。

## 5. 延时预算

| 环节 | 160 ms 块 | 240 ms 块 | 怎么省 |
|---|---|---|---|
| 攒一块 | 160 | 240 | 换更短的套（120 / 80 ms 也在 20 ms 格上，未测音质） |
| 每块计算（A4000，CUDA Graph） | 30 | 30 | 与块长无关；步数 10 → 6 省约 8 ms（未测音质） |
| 输出余量 | 50（锁频）/ 150 | 50 / 150 | 锁频 + 速率微调 |
| 算法固有（右前瞻 20 + 交叉淡化 40 + SOLA 搜索 20） | 80 | 80 | 交叉淡化 0.05 → 0.02 s 省 30 ms（导出参数，未测接缝） |
| 声卡进出缓冲 | 20–40 | 20–40 | WASAPI 独占 / 更小缓冲 |
| **合计** | **≈ 330–350** | **≈ 410–430** | |

## 6. 门控

- 默认 `SileroVad`：认语音本身，键盘、敲桌、翻纸不开门，干净语音全召回，起音延迟 0–160 ms。参数：阈值 0.5 开门，连续 2 块低于 0.35 关门，块电平低于 −60 dBFS 直接静音。CPU 每块约 1 ms，与推理线程同线程调用即可。
- `EnergyVad` 只看响度，连续打字会开门；仅作缺 Silero 模型时的退路。
- 门分不开的：别人的说话声（按设计放行）、部分咳嗽喷嚏。需要时加说话人相似度门（CAM++ 已在跑）或噪声增强微调，2.0 未做。
- 门关着时模型仍在算（窗口要滚），只是输出淡出到零；门一开模型立刻有正确的上下文。
- 离线文件转换不要挂门：剪好的文件开头就是语音，能量门的地板会钉在语音电平上，开头几秒全静音。

### 调门

三个参数是 `SileroVad` 的公开字段，使用方不改库也能调：`let mut gate = SileroVad::load(&rt, path)?; gate.threshold = 0.6;` 再传给 `Session::with_vad`。判决：块电平低于 `abs_floor_db` 直接静音；否则取块内各 32 ms 片的最大语音概率，门关着时超过 `threshold` 开门，门开着时连续 `hold_blocks` 块低于 0.7 × `threshold` 才关门。

| 参数（默认） | 调小 | 调大 |
|---|---|---|
| `threshold`（0.5） | 更不像语音的声音也开门：咳嗽、远处人声、带共振的敲击更易放行，怪词回来；轻声起音抓得更早 | 误开更少，但小声说话和词的第一个音素常不到位，掉词头；关门线随之上升，句中弱片段被当成安静，门翻转变多 |
| `hold_blocks`（2） | 1 时词间停顿就关门，门翻转暴涨，每次翻转一个 40 ms 斜坡，听感切字、吃词尾 | 4–5 时能跨过约 1 s 停顿，但句子说完后门仍开着约 1 s，这段里的键盘声原样进模型，怪词出现在句尾之后；翻转次数变少会掩盖真正的切话 |
| `abs_floor_db`（−60） | −80 等于关掉前置，全部交给 Silero 判，结果几乎不变，CPU 多约 1 ms | −45 就变回能量门：离麦克风远、增益低的说话被整块静音 |

调法：一次只动一个数，在 MimickingLab 里看两个量——门翻转次数（连续说话时应接近句子停顿数）和 `--gate-probe` 对 `C:/MimickingData/noise/` 四段录音的开门块数（键盘、敲桌应为 0）。

MimickingLab 里这三个数是可拖动的数值框（“求解与门控”组，选 Silero 时出现，改动即生效），无界面用 `--vad-threshold 0.5 --vad-hold 2 --vad-floor -60`（实时模式与 `--gate-probe` 都认）。

**新环境校准**（换房间、换麦克风或改了增益时）：

1. 用实际的麦克风和增益录两段 16 kHz 单声道 wav：30 s 只有该环境的背景音（键盘、空调、旁人说话都照常），30 s 自己正常说话（含几次停顿）。
2. `mimic-lab --headless --gate-probe bg.wav` 与 `--gate-probe speech.wav`，看 Silero 的开门块数：背景段应为 0（旁人说话按设计会放行，记下它占多少块），说话段应接近有声块数，每次起音最多晚 1 块。
3. 不合格就按上表改一个数（背景漏开 → 阈值 0.5 → 0.6；说话掉词头 → 阈值 0.4 或 hold 3；很安静的环境可以放到 0.4），加 `--vad-threshold / --vad-hold / --vad-floor` 重跑两段，直到两段都合格；再进实时模式听 1 分钟，看门翻转次数。
4. 把定下的三个数写进使用方自己的配置，启动时赋给 `SileroVad` 的字段；Lab 的数值框只影响当前进程。

## 7. 多路与容量

每路成本不随路数变（DiT 在此序列长度上是算力瓶颈），同一张卡各路顺序执行、每路延迟等于整轮：**每张 A4000 最多 4 路**（240 ms 块一轮约 115 ms），两卡 8 路。详见 `docs/capacity-a4000.md`。所有路加载完再开始；运行中加路要在块间隙创建会话（不重录 CUDA Graph，几毫秒）。

## 8. 验收

- 文件对拍：`mimic convert --reference ref.wav --source src.wav --out out.wav --models models/Mimicking-v2.0-lowlat`，输出应与 Lab 文件模式一致。
- 流式指标：MimickingLab `--headless --source S.wav --reference R.wav [--block 0.16] [--vad silero|energy|off]` 打印每块耗时分位、超时、门翻转、CAM++ 余弦；`--gate-probe FILE.wav` 单独看门控。
- 期望值（32 对 LibriTTS-R，流式 240 ms）：WER 7.6 %，英文 CAM++ 像目标 0.71 / 像源 0.42；每块 29–30 ms。
- 评测工具在 MimickingTrain：`tools/wer_eval.py`、`tools/pair_floor.py`、`tools/stream_diag.py`、`tools/noise_context.py`。

## 9. 常见问题（症状 → 原因 → 处理）

| 症状 | 原因 | 处理 |
|---|---|---|
| 延迟 2 秒以上 | 用了右前瞻 2.0 s 的早期套 | 用 `*-r0.02` 套 |
| 流式咬字糊、"拼音味"，离线正常 | 码本模型的图没冻结调节器统计量（2.0.0） | 用 2.0.1 图；自己导出必带 `--norm-stats` |
| 开头几秒静音（文件转换） | 离线挂了能量门 | 离线不挂门 |
| 敲键盘模型开口胡说 | 能量门 | Silero 门 |
| 每隔半分钟一下结巴 | 队列控制器丢输入 / 补静音对付时钟漂移 | 播放速率微调（§4） |
| 单路时每块耗时忽高忽低、欠载 | GPU 块间降频 | 锁频或增大余量 |
| 加载第二张卡时 CUDA error 901 | CUDA Graph 录制与别的 CUDA 调用并发 | 顺序加载（库已加锁，使用方别在加载期跑推理） |
| 换参考音色卡 350 ms | CAM++ 动态形状图不走 CUDA Graph | 在独立线程算 `prepare_prompt`，算好再 `set_prompt` |
| 说话时混着敲击声吃字 | 2.0 对叠加噪声敏感（模型限制） | 2.x 噪声增强微调；暂无运行时办法 |

## 10. 许可

库代码 Apache-2.0（`LICENSE`、`NOTICE`）。仓库不分发任何模型权重或导出的图；另行获取的模型按各自许可：2.0 的调节器与 DiT 权重为 MimickingTrain 自训；XLS-R、HiFT、CAM++ 为 Apache-2.0；Silero VAD 为 MIT（`models/vad/LICENSE-silero-vad.txt`）。逐项见 `docs/license-review.md`。
