English | [中文](README.zh-CN.md)

# History: the replication phase (October 2026)

Records written while the library was being built as a Rust replication of an existing real-time voice-conversion model, before the self-trained 2.0 weights existed. Numbers in these files are from the 0.1.0 era and are kept for the engineering record only; the current state is in `../release-2.0.md`.

| File | What it records |
|---|---|
| `rust-realtime-vc-plan.md` | The original plan: crate boundaries, milestones M0–M3, parity strategy |
| `m0-results.md` | M0: ONNX export and ONNX Runtime CUDA Graph validation (2026-10-26) |
| `m1-status.md` | M1: core library and offline conversion chain |
| `m2-status.md` | M2: device-resident binding, host DSP, multi-stream sessions, `mimic stream` |
| `m3-status.md` | M3: release preparation |
| `gpu-opt.md` | Export-side GPU optimisation: fp16 graphs, fused operators |
| `training-plan.md` | The training plan that led to MimickingTrain and the 1.0 / 2.0 weights |
