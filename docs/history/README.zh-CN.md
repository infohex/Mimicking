[English](README.md) | 中文

# 历史记录：复刻阶段（2026 年 10 月）

这些文档写于用 Rust 复刻一个现有实时变声模型的阶段，当时还没有自训的 2.0 权重。其中的数字都是 0.1.0 时代的，只作工程记录；现状见 `../release-2.0.md`。

| 文件 | 记录内容 |
|---|---|
| `rust-realtime-vc-plan.md` | 最初的总体计划：crate 边界、里程碑 M0–M3、对拍策略 |
| `m0-results.md` | M0：ONNX 导出与 ONNX Runtime CUDA Graph 验证（2026-10-26） |
| `m1-status.md` | M1：核心库与离线转换链 |
| `m2-status.md` | M2：设备驻留绑定、主机 DSP、多路会话、`mimic stream` |
| `m3-status.md` | M3：发布准备 |
| `gpu-opt.md` | 导出侧 GPU 优化：fp16 图、融合算子 |
| `training-plan.md` | 促成 MimickingTrain 与 1.0 / 2.0 权重的训练计划 |
