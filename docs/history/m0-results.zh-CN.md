[English](m0-results.md) | 中文

# M0 结果：ONNX 导出与 onnxruntime CUDA Graph 验证（2026-10-26）

环境：RTX A4000（GPU 1），onnxruntime-gpu 1.22.0，CUDA EP，torch 2.4.0+cu124 做参考。配置为 seed-vc 原版 GUI 默认（块 0.25 s，窗口 4.8 s，提示 3 s，10 步，CFG 0.7），序列 499 帧（258 提示 + 241 内容）。
脚本：`tools/export/export_all.py`（导出 + 夹具）、`validate_ort.py`（精度 + 计时）、`diag_placement.py`（节点分配）。原始数字在 `tools/export/validate_results.json`。

## 1. 验收门

| 门 | 目标 | 实测 | 结果 |
|---|---|---|---|
| DiT 单步（batch 2 含 CFG） | ≤ 5 ms | 4.95 ms（CUDA Graph）；eager 7.9 ms | 过，无余量 |
| 整块 10 步（GPU 图合计） | ≤ 80 ms | 70.7 ms；加主机侧 DSP 估 2 到 3 ms | 过 |
| batch 4（2 路）整块 | ≤ 120 ms | DiT 10 步 99 ms + XLSR、HiFT 按线性外推约 40 ms ≈ 140 ms | **不过，且门本身的前提错了，见 §3** |
| 各图精度 | 相对 L2 ≤ 1e-2 | 全部 ≤ 4e-3 | 过 |

结论：**M0 通过，可以进入 M1**，但方案 v3 里"合批几乎免费"的假设必须删掉。

## 2. 导出

| 图 | 输入 → 输出（静态形状） | 节点数（onnxsim 前 → 后） | 大小 | 备注 |
|---|---|---|---|---|
| xlsr.onnx | (1, 76800) f32 → (1, 239, 1024) f32 | 756 → 479 | 329 MB | 整体 fp16，归一化在图内 |
| length_regulator.onnx | (1, 139, 1024) → (1, 241, 384) | 65 → 43 | 9 MB | fp32，目标长度常量 |
| dit_step.onnx（另有 b4/b8/b16） | x, prompt_x (2, 80, 499), t (2,), style (2, 192), mu (2, 499, 384) → v (2, 80, 499) f16 | 2419 → 791 | 42 MB | autocast 导出，掩码与 RoPE 切片烘成常量 |
| campplus.onnx | (1, T, 80) → (1, 192) | 3102 | 28 MB | 动态 T，一次性使用，允许 CPU 回退 |
| hift_f0.onnx | (1, 80, 241) → (1, 241) | 45 → 17 | 7 MB | |
| hift_body.onnx | mel (1, 80, 241) + s_stft (1, 18, 15425) → (1, 18, 15425) | 1244 → 424 | 35 MB | 谐波源、16 点 STFT、ISTFT 在图外 |
| hift_source.npz | l_linear 权重、正弦源常数、窗 | | | 给 Rust 用 |

导出过程中踩到并解决的问题：
- 带动态 batch 轴导出时，Shape/Gather 子图被 ORT 放到 CPU，CUDA Graph 捕获被拒绝。改为全静态导出 + onnxsim 后，五张实时图全部落在 CUDA EP（`diag_placement.py` 用 `session.disable_cpu_ep_fallback` 验证）。
- 模型在 GPU 上时 torch 导出器的常量折叠报 cpu/cuda 混用，DiT 和 HiFT 关掉折叠，交给 onnxsim。
- autocast 导出的图输出是 fp16，绑定缓冲要按图的实际类型分配。

## 3. 计时（CUDA Graph，I/O 绑定到设备缓冲）

| 项 | eager | CUDA Graph | Python 版（bench/RESULTS.md） |
|---|---|---|---|
| XLSR | 8.2 ms | 6.2 ms | 11 到 23 ms |
| 长度调节 | 0.57 ms | 0.35 ms | 1.5 到 2.8 ms |
| DiT 单步 batch 2 | 7.9 ms | 4.95 ms | 14.2 ms（torch CUDA Graph 3.6 ms） |
| DiT 10 步含主机侧欧拉更新与拷贝 | | 52 到 56 ms | 137 到 155 ms |
| HiFT f0 | 2.1 ms | 0.41 ms | |
| HiFT 主体 | 16.6 ms | 14.3 ms | HiFT 整体 26 到 37 ms |
| **整块 GPU 合计，10 步** | | **70.7 ms** | 225 ms（带 VAD）/ 178 ms（不带） |
| 整块 GPU 合计，6 步 | | 50.9 ms | 约 170 ms（网关配置的实测） |

主机侧 DSP（numpy，供参考，不代表 Rust）：ISTFT 向量化后 2.2 ms；谐波源 + STFT 26.6 ms，主要是 numpy 对 55 万点的 cumsum/sin/随机数和 1.5 万帧小矩阵乘的开销，Rust 预计 1 到 3 ms。

### DiT batch 扫描（关键发现）

| batch（含 CFG 翻倍） | 路数 | 单步 ms | 每路每步 ms |
|---|---|---|---|
| 2 | 1 | 4.96 | 4.96 |
| 4 | 2 | 9.92 | 4.96 |
| 8 | 4 | 21.2 | 5.3 |
| 16 | 8 | 41.9 | 5.2 |

**线性**。之前在 PyTorch eager 下看到的"batch 1 和 2 耗时相同"是 14 ms 的调度开销把 2 到 4 ms 的真实计算盖住了。去掉调度开销后，DiT 在这个序列长度上就是算力瓶颈：25.7M 参数 × 1002 个 token ≈ 50 GFLOP/步，A4000 fp16 实际吞吐约 10 TFLOPS，正好 5 ms。推论：
- 合批对吞吐没有收益，只是调度上更简单；每路成本固定约 5 ms/步。
- CFG 关掉就真的减半（Python 版测不出来也是同一原因）。
- 每路每块的 GPU 预算：10 步约 71 ms，6 步约 51 ms；块 0.25 s 一张 A4000 约 3 路（10 步）或 4 路（6 步），块 0.3 s 再多一路。
- 想再快只能减算力：步数蒸馏、CFG 蒸馏、int8，或者更短的提示。

### ORT 图 vs torch 图

ORT 的 DiT 单步 4.95 ms 比 torch CUDA Graph 的 3.6 ms 慢 37%。原因估计是 autocast 导出留下的大量 Cast 节点和 SDPA 被拆成 MatMul/Softmax，没有融合注意力。可能的改善：导出时把权重直接存成 fp16 去掉 Cast；ORT 的 transformer 优化器（`onnxruntime.transformers`）做注意力融合；TensorRT EP。本机没装 TensorRT 运行库（缺 nvinfer_10.dll），`--trt` 未能测。这些留到 M1 之后作为优化项。

## 4. 精度（ORT 对 PyTorch fp16 夹具）

| 图 | max abs | mean abs | 相对 L2 |
|---|---|---|---|
| XLSR | 0.037 | 3.3e-4 | 4.0e-3 |
| 长度调节 | 6.0e-5 | 7.7e-6 | 2.4e-4 |
| CAM++ | 4.6e-3 | 9.4e-4 | 1.2e-3 |
| DiT 单步 | 0.014 | 2.0e-3 | 8.8e-4 |
| 10 步 CFM 输出梅尔谱（同一噪声） | 0.017 | 2.1e-3 | 4.0e-4 |
| HiFT f0（Hz） | 0.25 | 0.012 | 2.0e-4 |
| HiFT 主体 | 0.031 | 4.6e-4 | 3.2e-4 |
| numpy STFT vs torch | 8.9e-8 | | 8.2e-8 |
| numpy ISTFT vs torch | 3.6e-4 | 1.4e-5 | 5.0e-4 |
| 整条 HiFT（ORT 主体 + numpy ISTFT，用 torch 的激励源）输出波形 | 3.0e-3 | 4.9e-5 | 2.2e-3 |
| 同上的对数梅尔谱 | 0.25 | 6.2e-3 | 1.5e-3 |
| 整条 HiFT，自己生成随机激励源，对数梅尔谱 | 2.26 | 0.11 | 2.2e-2 |

最后一行的差异来自谐波源的随机相位和噪声（基线每次运行也不同），不是实现错误；用相同激励源时差异在 fp16 噪声范围内。M1 的 Rust 对拍以"相同激励源"为准，随机源那一路只看听感和说话人相似度。

## 5. 对方案 v3 的修正

1. 删除"合批几乎免费"及其派生结论。`BatchRunner` 保留，但定位是调度简化和减少图运行次数，不是吞吐倍增。
2. 性能门改为**每路**成本：单路整块 10 步 ≤ 80 ms（已达 71），6 步 ≤ 60 ms（已达 51）；路数 = 块长 × 0.6 / 每路整块。
3. 新增优化项（M1 之后）：fp16 权重直出去 Cast、ORT 注意力融合、TensorRT EP；预期把 DiT 单步从 4.95 ms 拉回 3.6 ms 附近。
4. Rust 侧 DSP 的实现依据已经固定：16 点 DFT 用矩阵乘而不是 FFT，ISTFT 用 4 条步进通道做重叠相加，都已在 numpy 里对拍到 1e-4 以下。

## 6. 产物

- `models/onnx/`：七张图 + `hift_source.npz`，约 570 MB，不入库。
- `tools/export/fixtures/`：每级的真实输入与 PyTorch 输出（`.npz`）和 `shapes.json`，是 M1 Rust 对拍测试的夹具。
- `tools/export/validate_results.json`：本页全部数字。
