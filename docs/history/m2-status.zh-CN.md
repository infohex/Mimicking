[English](m2-status.md) | 中文

# M2 状态：设备驻留绑定 + CUDA Graph、主机 DSP、多路会话、`mimic stream`（2026-10-26）

## 结论

**M2 完成。** 单路每块从 M1 的 147 ms 降到 **82 ms（p50）/ 85 ms（p95）**，PyTorch 流式版为 199 ms（p95 265）。实时节拍下单路 102 块零超时。一张 A4000 上 10 步可带 2 路，6 步可带 3 路（块 0.25 s）。验收门"10 步 p95 ≤ 80 ms"差 5 ms，原因和余地见 §4。

## 做了什么

| 项 | 实现 | 收益 |
|---|---|---|
| 设备驻留 I/O 绑定 | `mimic-onnx::BoundGraph`：每张静态图的输入输出都绑定到 ORT 分配的 CUDA 缓冲，主机拷贝走 cudart（`cudarc` 运行时加载，`cudaMemcpyDefault`） | 单独看不大（147 → 137 ms），但它是 CUDA Graph 的前提 |
| CUDA Graph | `OnnxRuntime::with_cuda_graph(true)`，加载后预跑两次完成捕获；动态形状的 CAM++ 不开 | 137 → 106 ms；DiT 单步 7.9 → 4.95 ms |
| 主机 DSP | HiFT 谐波源改为每帧一次 sin/cos 后相量旋转、9 路噪声合成一路高斯；STFT/ISTFT 用切片和 DFT 矩阵 | vocode 主机部分 30 ms → 7.7 ms（源 2.0、STFT 2.4、ISTFT 3.3） |
| 求解器上下文 | `Graph::set_input`/`run_set`：prompt_x、style、mu 每块传一次，每步只传 x 和 t | 约 2 ms |
| 融合步进 | `dit_step_fused.onnx` 把欧拉更新和 CFG 合并进图，x 常驻设备，每步只传 t/dt/cfg，输出设备内拷回输入（`copy_output_to_input`） | 约 1 ms；更大的意义是每步不再有主机往返，为以后合批和异步留路 |
| 多路 | `Runner`（持有模型）+ `Session`（窗口、SOLA、门控、RNG、提示），同一 Runner 轮询任意数量 Session | |
| VAD | `mimic-dsp::EnergyVad`：绝对门限 −48 dBFS、自适应底噪、开 +12 dB、关 +6 dB 持续 1 块，输出一块斜坡 | 静音段不"编话" |
| `mimic stream` | N 个源文件 = N 路，可按块周期节拍（`--realtime`）或全速跑，统计每路 p50/p95/max、整轮耗时、超时次数 | |

## 测量（A4000，GPU 1，块 0.25 s = 240 ms 周期，CFG 0.7）

### 单路

| | 每块 p50 | p95 | content | solve | vocode |
|---|---|---|---|---|---|
| M1 eager | 147 | | 8.4 | 87.2 | 50.2 |
| 绑定、无 CUDA Graph | 136.5 | | 8.0 | 80.4 | 47.2 |
| CUDA Graph | 106 | | 6.1 | 54.6 | 44.6 |
| + 主机 DSP | 88 | | 6.2 | 54.6 | 26.8 |
| + 上下文/融合步进 | **81.9** | **85.1** | 6.3 | 52.0 | 23.0 |
| PyTorch 流式参考 | 199 | 265 | | | |

实时节拍（`--realtime --vad`，22 s 源）：p50 90 ms、p95 107 ms、102 块零超时。节拍下比全速高约 8 ms，是 GPU 在块间空闲后降频，服务器上多路并发时不会出现。

### 多路（全速，12 s 源，每轮 = 所有路各推一块）

| 路数 | 步数 | 每轮 p50 | 对 240 ms 周期 |
|---|---|---|---|
| 1 | 10 | 82 | 过 |
| 3 | 10 | 247 | 不过 |
| 3 | 6 | 189 | 过 |
| 4 | 6 | 249 | 不过 |

规则：路数 ≈ 周期 × 0.8 / 每路整块。每路成本不随路数变化（DiT 是算力瓶颈，M0 已证明合批无收益），所以这就是这张卡在这个模型上的容量。

### 质量

| 输出 | CAM++ 余弦（对参考） |
|---|---|
| `mimic convert`，融合步进 | 0.661 |
| `mimic stream --vad`，22 s | 0.669 |
| PyTorch 流式参考 | 0.647 |
| 源对参考基线 | 0.27 / 0.12 |

七项 ORT 对拍（含融合步进 10 步）与六项 dsp 对拍全部通过，数字与 M1 相同。

## 4. 没达到的和余地

- 10 步 p95 85 ms 对门 80 ms。构成：GPU 图合计 70.7（XLSR 6.2、DiT 49.5、HiFT 14.7）+ 主机 DSP 7.7 + 拷贝和求解器开销约 4。要过门只能动 GPU 部分：fp16 权重直出去 Cast、ORT 注意力融合、TensorRT EP（M0 已列），预期 DiT 单步 4.95 → 3.6 ms 即可省 13 ms。这些是导出侧的工作，不改 Rust。
- 块长是图的形状参数：换块长要重新导出一套图（`export_all.py --block`）放到另一个目录。CLI 对形状不符现在会给出明确提示。
- 噪声门是能量门，够用但简单；trait 已定，可换。
- 合批（batch N 图）没做：M0 证明没有吞吐收益，只剩调度上的简化，暂不值得。

## 5. 文件

- `crates/mimic-onnx`：`BoundGraph`、`DeviceBuf`（cudart 拷贝）、`Graph` 新增 `set_input/run_set/run_set_keep/get_output/copy_output_to_input`
- `crates/mimic-core`：`Vad`、`FlowStepper`
- `crates/mimic-dsp`：`vad.rs`、重写的 `hift.rs`、`noise.rs`（Box–Muller 对）
- `crates/mimic-model`：`DitStepFused`、`load_static`（CUDA 下走 BoundGraph 并预热）
- `crates/mimic-pipeline`：`Estimator` 枚举、`Session`/`Runner`、`solve_fused`
- `crates/mimic-cli`：`stream` 子命令、`--no-cuda-graph`
- `tools/export/export_all.py`：`DiTStepFused` 导出
