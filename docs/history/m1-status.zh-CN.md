[English](m1-status.md) | 中文

# M1 状态：Rust 核心库 + 离线转换链（2026-10-26）

## 结论

**M1 验收通过。** 六个 crate 编译通过；信号处理和五张图逐级对拍全部在容差内；`mimic convert` 端到端产出的音频，说话人相似度与 PyTorch 流式版相同。性能尚未优化（ORT eager、主机侧拷贝、未开 CUDA Graph），属于 M2 的内容。

## 代码结构

```
Cargo.toml                 workspace（ndarray 0.17、ort 2.0.0-rc.15 以 load-dynamic + api-22 对接 onnxruntime 1.22）
crates/
  mimic-core      trait：Graph、ContentEncoder、SpeakerEncoder、LengthRegulator、VelocityEstimator、Vocoder、MelAnalyzer、Resampler；Prompt
  mimic-dsp       SincResampler（torchaudio 算法）、MelSpec（seed-vc 梅尔谱）、KaldiFbank（CAM++ 输入）、hift（谐波源、16 点 STFT/ISTFT）、Sola、noise
  mimic-onnx      OnnxRuntime（运行时加载 dylib、CUDA EP）、OnnxGraph（实现 Graph）
  mimic-model    WindowConfig/Shapes（窗口几何，含 numpy 的"五成双"取整）、XlsrEncoder、InterpolateRegulator、CampplusSpeaker、DitStep、HiftVocoder、ModelSet
  mimic-pipeline  EulerCfg 求解器、Stages、prepare_prompt、StreamSession（滑窗 + SOLA）、Converter（文件通过流式会话）
  mimic-cli       mimic convert --reference --source --out [--steps --cfg --block --extra-right --seed]
tools/export/     export_all.py（七张静态图 + 夹具）、validate_ort.py、diag_placement.py、py_stream_convert.py（PyTorch 流式参考输出）、eval_outputs.py（CAM++ 余弦）
```

设计上与基线不同的点已落实：单步去噪图 + Rust 欧拉求解器；Prompt 是显式对象；流式状态机与音频 I/O、进程边界无关；核心 crate 不含任何使用方知识。

## 逐级对拍（Rust 对 PyTorch 夹具）

`cargo test -p mimic-dsp --release`（纯 Rust，无需 GPU）：

| 模块 | max abs | 相对 L2 |
|---|---|---|
| 重采样 22.05 k → 16 k（torchaudio sinc） | 4.2e-6 | |
| 梅尔谱（22.05 k / 1024 / 256 / 80） | 2.1e-4 | 6.9e-7 |
| Kaldi fbank（均值归一） | 3.4e-4 | 3.7e-6 |
| HiFT 16 点 STFT | 1.5e-7 | |
| HiFT ISTFT → 波形 | 3.6e-4 | 5.0e-4 |
| 谐波源 RMS | 0.0282 对 0.0282 | 随机相位，只比能量 |

`MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=1 cargo test -p mimic-model --release`（PATH 需含 torch/lib 以找到 cuDNN）：

| 阶段 | max abs | 相对 L2 | 备注 |
|---|---|---|---|
| XLSR 图 | 3.8e-2 | 3.9e-3 | fp16 |
| 长度调节图 | 6.0e-5 | 2.4e-4 | |
| CAM++（Rust fbank → 图） | 3.6e-3 | 1.1e-3 | 余弦 1.000000 |
| DiT 单步图 | 1.4e-2 | 8.8e-4 | |
| 欧拉 10 步 + CFG（Rust 求解器） | 1.7e-2 | 4.0e-4 | 与 Python ORT 验证数字一致 |
| HiFT 整链（相同激励源） | 2.6e-3 | 2.0e-3 | |

## 端到端

同一源（`source_s1.wav`，12.5 s）、同一参考（`azuma_0.wav`）、GUI 默认几何、10 步、CFG 0.7，GPU 1：

| | Rust `mimic convert` | PyTorch 流式参考（`py_stream_convert.py`） |
|---|---|---|
| CAM++ 余弦（输出 vs 参考） | **0.661** | 0.647 |
| CAM++ 余弦（输出 vs 源） | 0.691 | 0.690 |
| 源 vs 参考基线 | 0.266 | |
| 输出电平 | −24.1 dBFS | −24.1 dBFS |
| 两份输出的对数梅尔距离 | 0.66（同一输出自身移一帧为 0.59，即差异在随机噪声量级） | |
| 每块耗时（62 块） | 147 ms（content 8.4、regulate 1.2、solve 87.2、vocode 50.2） | 中位 199 ms、p95 265 ms |
| RTF | 0.73 | 约 0.8 |

输出文件：`out/rust_azuma_s1.wav`、`out/py_azuma_s1.wav`。

## 实现中踩到的坑

- 带动态 batch 轴导出会让形状算子落到 CPU，CUDA Graph 捕获被拒；全静态 + onnxsim 解决。XLSR 为块窗口和 3 s 参考各导一张（`xlsr_ref.onnx`，权重重复 329 MB，以后可共享权重）。
- autocast 导出的图输出 fp16；导出脚本统一在图末加 Cast，Rust 侧只处理 f32。
- `np.round` 是五成双：0.25 s 块 = 12.5 单位 → 12；Rust 要用 `round_ties_even`，否则序列长度 502 对 499。
- Windows 上 numpy 1.x 默认整数是 int32，npz 标量读取要兼容 i64/i32/f64。
- `ort` rc.15 的 `GraphOptimizationLevel::Level3` 被 onnxruntime 1.22 拒绝，用默认即可。
- rand 0.9 没有 `StandardNormal`，自带 Box–Muller。

## M2 要做的性能项（当前 147 ms → 目标 ≤ 80 ms）

1. 设备驻留 I/O 绑定 + CUDA Graph（ORT 验证里 DiT 单步 7.9 → 4.95 ms，整块 GPU 合计 70.7 ms）。`ort` 有 `IoBinding`、`Allocator`、`Tensor::new(device)`，缺的是主机到设备的拷贝接口，可能要用 CUDA 驱动 API（cudarc）做 memcpy，或把欧拉更新也放进图里。
2. 主机侧 DSP：谐波源每样本 9 次 Box–Muller 和 ndarray 索引是 vocode 50 ms 里的大头；预生成噪声、去掉边界检查。
3. 批量/多路、VAD 门控、线程模型。
