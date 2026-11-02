English | [中文](m0-results.zh-CN.md)

# M0 results: ONNX export and onnxruntime CUDA Graph validation (2026-10-26)

Environment: RTX A4000 (GPU 1), onnxruntime-gpu 1.22.0, CUDA EP, torch 2.4.0+cu124 as reference. Configuration is the original seed-vc GUI default (block 0.25 s, window 4.8 s, prompt 3 s, 10 steps, CFG 0.7), sequence 499 frames (258 prompt + 241 content).
Scripts: `tools/export/export_all.py` (export + fixtures), `validate_ort.py` (accuracy + timing), `diag_placement.py` (node placement). Raw numbers in `tools/export/validate_results.json`.

## 1. Acceptance gates

| Gate | Target | Measured | Result |
|---|---|---|---|
| DiT single step (batch 2 with CFG) | ≤ 5 ms | 4.95 ms (CUDA Graph); eager 7.9 ms | Pass, no margin |
| Whole block, 10 steps (GPU graphs total) | ≤ 80 ms | 70.7 ms; plus host-side DSP estimated 2 to 3 ms | Pass |
| Batch 4 (2 streams) whole block | ≤ 120 ms | DiT 10 steps 99 ms + XLSR, HiFT extrapolated linearly about 40 ms ≈ 140 ms | **Fail, and the premise of the gate itself was wrong, see §3** |
| Accuracy of each graph | relative L2 ≤ 1e-2 | all ≤ 4e-3 | Pass |

Conclusion: **M0 passes, M1 can start**, but the assumption in plan v3 that "batching is almost free" must be removed.

## 2. Export

| Graph | Input → output (static shapes) | Node count (before → after onnxsim) | Size | Notes |
|---|---|---|---|---|
| xlsr.onnx | (1, 76800) f32 → (1, 239, 1024) f32 | 756 → 479 | 329 MB | fp16 throughout, normalisation inside the graph |
| length_regulator.onnx | (1, 139, 1024) → (1, 241, 384) | 65 → 43 | 9 MB | fp32, target length is a constant |
| dit_step.onnx (also b4/b8/b16) | x, prompt_x (2, 80, 499), t (2,), style (2, 192), mu (2, 499, 384) → v (2, 80, 499) f16 | 2419 → 791 | 42 MB | autocast export, masks and RoPE slices baked into constants |
| campplus.onnx | (1, T, 80) → (1, 192) | 3102 | 28 MB | dynamic T, used once, CPU fallback allowed |
| hift_f0.onnx | (1, 80, 241) → (1, 241) | 45 → 17 | 7 MB | |
| hift_body.onnx | mel (1, 80, 241) + s_stft (1, 18, 15425) → (1, 18, 15425) | 1244 → 424 | 35 MB | harmonic source, 16-point STFT and ISTFT outside the graph |
| hift_source.npz | l_linear weights, sine source constants, window | | | for the Rust side |

Problems hit and solved during export:
- When exporting with a dynamic batch axis, ORT placed the Shape/Gather subgraphs on the CPU and CUDA Graph capture was refused. After switching to fully static export + onnxsim, all five real-time graphs land on the CUDA EP (verified by `diag_placement.py` with `session.disable_cpu_ep_fallback`).
- With the model on the GPU, the torch exporter's constant folding reports mixed cpu/cuda; folding is turned off for DiT and HiFT and left to onnxsim.
- Graphs exported under autocast output fp16; bound buffers must be allocated according to the graph's actual type.

## 3. Timing (CUDA Graph, I/O bound to device buffers)

| Item | eager | CUDA Graph | Python version (bench/RESULTS.md) |
|---|---|---|---|
| XLSR | 8.2 ms | 6.2 ms | 11 to 23 ms |
| Length regulator | 0.57 ms | 0.35 ms | 1.5 to 2.8 ms |
| DiT single step batch 2 | 7.9 ms | 4.95 ms | 14.2 ms (torch CUDA Graph 3.6 ms) |
| DiT 10 steps including host-side Euler update and copies | | 52 to 56 ms | 137 to 155 ms |
| HiFT f0 | 2.1 ms | 0.41 ms | |
| HiFT body | 16.6 ms | 14.3 ms | HiFT overall 26 to 37 ms |
| **Whole block GPU total, 10 steps** | | **70.7 ms** | 225 ms (with VAD) / 178 ms (without) |
| Whole block GPU total, 6 steps | | 50.9 ms | about 170 ms (measured with the gateway configuration) |

Host-side DSP (numpy, for reference, not representative of Rust): ISTFT 2.2 ms after vectorisation; harmonic source + STFT 26.6 ms, mostly numpy overhead for cumsum/sin/random numbers over 550 k samples and 15 k small matrix multiplies per frame; Rust expected at 1 to 3 ms.

### DiT batch sweep (key finding)

| Batch (doubled for CFG) | Streams | Per step ms | Per stream per step ms |
|---|---|---|---|
| 2 | 1 | 4.96 | 4.96 |
| 4 | 2 | 9.92 | 4.96 |
| 8 | 4 | 21.2 | 5.3 |
| 16 | 8 | 41.9 | 5.2 |

**Linear.** The "batch 1 and 2 take the same time" seen earlier under PyTorch eager was 14 ms of scheduling overhead hiding 2 to 4 ms of real compute. With the scheduling overhead removed, DiT at this sequence length is simply compute-bound: 25.7M parameters × 1002 tokens ≈ 50 GFLOP/step, A4000 real fp16 throughput about 10 TFLOPS, which is exactly 5 ms. Consequences:
- Batching gives no throughput gain, only simpler scheduling; per-stream cost is fixed at about 5 ms/step.
- Turning CFG off really does halve it (the Python version could not measure this for the same reason).
- GPU budget per stream per block: about 71 ms at 10 steps, about 51 ms at 6 steps; with a 0.25 s block one A4000 takes about 3 streams (10 steps) or 4 streams (6 steps), one more with a 0.3 s block.
- Going faster means cutting compute: step distillation, CFG distillation, int8, or a shorter prompt.

### ORT graph vs torch graph

ORT's DiT single step at 4.95 ms is 37% slower than torch CUDA Graph at 3.6 ms. The likely cause is the many Cast nodes left by the autocast export and SDPA being split into MatMul/Softmax with no fused attention. Possible improvements: store the weights directly as fp16 at export to remove the Casts; the ORT transformer optimiser (`onnxruntime.transformers`) for attention fusion; the TensorRT EP. The TensorRT runtime libraries are not installed on this machine (nvinfer_10.dll missing), so `--trt` could not be tested. These are left as optimisation items after M1.

## 4. Accuracy (ORT against PyTorch fp16 fixtures)

| Graph | max abs | mean abs | relative L2 |
|---|---|---|---|
| XLSR | 0.037 | 3.3e-4 | 4.0e-3 |
| Length regulator | 6.0e-5 | 7.7e-6 | 2.4e-4 |
| CAM++ | 4.6e-3 | 9.4e-4 | 1.2e-3 |
| DiT single step | 0.014 | 2.0e-3 | 8.8e-4 |
| 10-step CFM output mel spectrogram (same noise) | 0.017 | 2.1e-3 | 4.0e-4 |
| HiFT f0 (Hz) | 0.25 | 0.012 | 2.0e-4 |
| HiFT body | 0.031 | 4.6e-4 | 3.2e-4 |
| numpy STFT vs torch | 8.9e-8 | | 8.2e-8 |
| numpy ISTFT vs torch | 3.6e-4 | 1.4e-5 | 5.0e-4 |
| Full HiFT (ORT body + numpy ISTFT, with torch's excitation source), output waveform | 3.0e-3 | 4.9e-5 | 2.2e-3 |
| Same, log mel spectrogram | 0.25 | 6.2e-3 | 1.5e-3 |
| Full HiFT, own random excitation source, log mel spectrogram | 2.26 | 0.11 | 2.2e-2 |

The difference in the last row comes from the random phase and noise of the harmonic source (the original seed-vc also differs from run to run), not from an implementation error; with the same excitation source the difference is within fp16 noise. The M1 Rust parity uses "same excitation source" as the reference; the random-source path is judged only by listening and speaker similarity.

## 5. Corrections to plan v3

1. Remove "batching is almost free" and the conclusions derived from it. `BatchRunner` stays, but its role is scheduling simplification and fewer graph runs, not a throughput multiplier.
2. The performance gate becomes a **per-stream** cost: single stream whole block 10 steps ≤ 80 ms (71 reached), 6 steps ≤ 60 ms (51 reached); streams = block length × 0.6 / per-stream whole block.
3. New optimisation items (after M1): fp16 weights emitted directly to drop the Casts, ORT attention fusion, TensorRT EP; expected to bring the DiT single step from 4.95 ms back to around 3.6 ms.
4. The basis for the Rust-side DSP implementation is fixed: the 16-point DFT uses a matrix multiply rather than an FFT, the ISTFT uses 4 strided channels for overlap-add, both already matched in numpy to below 1e-4.

## 6. Deliverables

- `models/onnx/`: seven graphs + `hift_source.npz`, about 570 MB, not checked in.
- `tools/export/fixtures/`: real inputs and PyTorch outputs for each stage (`.npz`) and `shapes.json`; these are the fixtures for the M1 Rust parity tests.
- `tools/export/validate_results.json`: every number on this page.
