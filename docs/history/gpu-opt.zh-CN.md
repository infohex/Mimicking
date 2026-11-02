[English](gpu-opt.md) | 中文

# 导出侧 GPU 优化：fp16 直出、融合算子（2026-10-26）

## 结论

**DiT 单步 4.95 → 2.10 ms（CUDA Graph），目标 3.6 ms 超额完成。** HiFT 主体 14.3 → 10.2 ms。单路整块从 82 ms 降到 **48 ms p50 / 53 ms p95**（门 80 ms），PyTorch 流式版为 199 / 265。一张 A4000 上 10 步可带 4 路、6 步可带 5 路（块 0.25 s）。

## 诊断

`tools/export/profile_graph.py` 看 autocast 导出的 `dit_step.onnx`：792 个节点，Gemm/MatMul 只有 75 个，其余是 Cast 93、Reshape 106、Mul 169、Transpose 48、Unsqueeze 57、Gather 36、Tile 18（`repeat_interleave(1)` 的空操作）。CUDA Graph 消除了 CPU 下发，但每个核函数仍有几微秒的 GPU 侧固定成本，约 790 个核函数就是 2 ms 以上的纯浪费。所以目标不是"更快的算子"，而是"更少的算子"。

## DiT：按同一套权重重写前向（`tools/export/export_dit_opt.py`）

导出包装器 `DiTClean` 不再调用 seed-vc 原版的 `forward`，而是用基线的权重按同样的数学重写，并通过 `torch.autograd.Function.symbolic` 直接发射 ORT 的融合算子：

| seed-vc 原版写法 | 发射的算子 | 每层节省 |
|---|---|---|
| RMSNorm：Mul、ReduceMean、Add、Sqrt、Div、Mul、Cast | `SimplifiedLayerNormalization`（ORT 默认域内部算子，内部 fp32 累加） | 约 6 |
| RoPE：reshape 成对、4 个 Mul、Stack、Flatten | `com.microsoft::RotaryEmbedding(interleaved=1)`，cos/sin 表烘成常量 | 约 10 × 2 |
| SDPA 展开：Transpose ×4、MatMul ×2、Softmax、Tile ×2 | `com.microsoft::MultiHeadAttention(num_heads=6)` | 约 9 |
| SiLU：Sigmoid、Mul | `com.microsoft::QuickGelu(alpha=1)` | 1 |
| autocast 的 93 个 Cast | 整图 fp16，只在 5 个输入和 1 个输出处 Cast | 87 |

结果 190 个节点（MatMul 51、Add 25、SimplifiedLayerNormalization 19、RotaryEmbedding 18、MultiHeadAttention 9、QuickGelu 9）。融合版（欧拉更新在图内）201 个。

两个必须注意的精度点：
- 时间步正弦表 `1000·t·freq` 最高约 1000 弧度，fp16 的分辨率是 0.5 弧度，整模型 `.half()` 会把这张表毁掉；包装器在 `.half()` 之前保存 fp32 副本，嵌入计算在 fp32 做完再转 fp16。
- RoPE 的 cos/sin 缓存必须与激活同为 fp16，否则 ORT 报类型参数不一致。

导出时 ONNX 检查器不认识默认域里的 `SimplifiedLayerNormalization`，用 `OperatorExportTypes.ONNX_FALLTHROUGH` 跳过检查。

| | 节点 | eager | CUDA Graph | 对夹具相对 L2 |
|---|---|---|---|---|
| autocast 导出 | 792 | 8.3 ms | 4.95 ms | 8.8e-4 |
| 重写 + 融合 + fp16 | 190 | 3.0 ms | **2.10 ms** | 9.4e-4 |
| 重写的 fp32 版（只验证重写正确） | | | | 7.1e-4 |

融合图 10 步（Python 循环含主机拷贝）22.5 ms，输出梅尔谱相对 L2 3.8e-4。

## HiFT 主体：纯 fp16 + 防溢出的 Snake（`tools/export/export_misc_opt.py`）

autocast 版 425 个节点里有大量 Cast。直接 `.half()` 导出后 NaN：Snake 激活 `x + sin²(αx)/(α+1e-9)` 的 α 是线性刻度、取值 −2.48 到 2.30、会过零，fp16 下 `1e-9` 消失、`1/α` 溢出。改为数学等价的 `x + sign(α)·(sin(αx)·|α|^-½)²`，`|α|^-½` 在 fp32 预算（最大约 3.2e4，在 fp16 范围内），中间量全部有界。

| | 节点 | eager | CUDA Graph | 相对 L2 |
|---|---|---|---|---|
| autocast | 425 | 16.2 ms | 14.26 ms | 3.2e-4 |
| fp16 + 安全 Snake | 372 | 11.9 ms | **10.17 ms** | 6.8e-4 |

## XLSR：放弃

ORT transformers 优化器（bert 模式）只匹配到 GELU 和 SkipLayerNorm，没匹配 wav2vec2 的注意力模式；CUDA Graph 6.20 → 5.96 ms，而且 `convert_float_to_float16` 后精度崩（相对 L2 1.12，估计是特征编码器的归一化或位置卷积溢出）。6 ms 里值得追的只有注意力那部分，需要像 DiT 一样手写包装器发射 MHA，收益约 1 到 2 ms，暂不做。

## 整块与多路（Rust，`mimic`，A4000，块 0.25 s = 240 ms 周期）

| | 每块 p50 | p95 | content | solve | vocode |
|---|---|---|---|---|---|
| M2 结束时 | 81.9 | 85.1 | 6.3 | 52.0 | 23.0 |
| + DiT 优化图 | 53.7 | 59.2 | 6.3 | 23.2 | 23.9 |
| + HiFT 优化图 | **47.8** | **53.1** | 6.3 | 23.0 | 19.1 |

| 路数 | 步数 | 每轮 p50 / p95 | 超时 |
|---|---|---|---|
| 3 | 10 | 158 / 168 | 0 |
| 4 | 10 | 194 / 206 | 0 |
| 5 | 6 | 222 / 235 | 0 |
| 6 | 6 | 242 / 257 | 26/42 |

七项 ORT 对拍全部通过（DiT 单步 9.4e-4，10 步 3.8e-4，HiFT 整链 3.0e-3）；端到端 CAM++ 余弦 0.669（PyTorch 0.647）。

**一个运维上的现象**：实时节拍单路（每 240 ms 才忙 48 ms）测得 p50 94 ms，几乎是全速的两倍，content 和 solve 都翻倍。这是 GPU 在块间空闲后降频，不是代码问题；多路并发时 GPU 持续忙就没有。单路部署要稳定延迟可以锁频（`nvidia-smi -lgc`）或保持 GPU 常驻负载。

## 文件与流程

- 导出流程：`export_all.py`（基础图 + 夹具）→ `export_dit_opt.py --install`（DiT 两张）→ `export_misc_opt.py --skip-xlsr --install`（HiFT 主体）。安装时原 autocast 图备份为 `*.autocast.onnx`，优化图另存 `*_opt.onnx`。
- 剩余构成（48 ms）：DiT 10 步 21 + XLSR 6.2 + HiFT 10.6 + 主机 DSP 7.7 + 拷贝与调度约 2。再往下的选项：XLSR 手写 MHA 包装器（约 1 到 2 ms）、主机 DSP SIMD（约 3 ms）、步数蒸馏（模型层）。
