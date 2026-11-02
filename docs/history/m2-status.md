English | [中文](m2-status.zh-CN.md)

# M2 status: device-resident binding + CUDA Graph, host DSP, multi-stream sessions, `mimic stream` (2026-10-26)

## Conclusion

**M2 complete.** Single-stream per-block time dropped from M1's 147 ms to **82 ms (p50) / 85 ms (p95)**; the PyTorch streaming version is 199 ms (p95 265). Under real-time pacing a single stream ran 102 blocks with zero overruns. One A4000 carries 2 streams at 10 steps and 3 streams at 6 steps (block 0.25 s). The acceptance gate "10 steps p95 ≤ 80 ms" is missed by 5 ms; the reasons and the remaining headroom are in §4.

## What was done

| Item | Implementation | Gain |
|---|---|---|
| Device-resident I/O binding | `mimic-onnx::BoundGraph`: the inputs and outputs of every static graph are bound to ORT-allocated CUDA buffers, host copies go through cudart (`cudarc` loaded at runtime, `cudaMemcpyDefault`) | Small on its own (147 → 137 ms), but it is the prerequisite for CUDA Graph |
| CUDA Graph | `OnnxRuntime::with_cuda_graph(true)`, two warm-up runs after loading complete the capture; not enabled for CAM++ with its dynamic shapes | 137 → 106 ms; DiT single step 7.9 → 4.95 ms |
| Host DSP | HiFT harmonic source changed to one sin/cos per frame followed by phasor rotation, 9 noise channels merged into one Gaussian; STFT/ISTFT use slicing and DFT matrices | vocode host part 30 ms → 7.7 ms (source 2.0, STFT 2.4, ISTFT 3.3) |
| Solver context | `Graph::set_input`/`run_set`: prompt_x, style, mu are uploaded once per block, each step uploads only x and t | about 2 ms |
| Fused step | `dit_step_fused.onnx` merges the Euler update and CFG into the graph, x stays resident on the device, each step uploads only t/dt/cfg, the output is copied back to the input on the device (`copy_output_to_input`) | about 1 ms; the larger point is that there is no host round trip per step any more, which leaves room for later batching and async |
| Multi-stream | `Runner` (holds the models) + `Session` (window, SOLA, gating, RNG, prompt); one Runner polls any number of Sessions | |
| VAD | `mimic-dsp::EnergyVad`: absolute threshold −48 dBFS, adaptive noise floor, open at +12 dB, close at +6 dB held for 1 block, output ramps over one block | Silent segments no longer "make up speech" |
| `mimic stream` | N source files = N streams, run paced to the block period (`--realtime`) or at full speed; reports per-stream p50/p95/max, time per round, overrun count | |

## Measurements (A4000, GPU 1, block 0.25 s = 240 ms period, CFG 0.7)

### Single stream

| | Per-block p50 | p95 | content | solve | vocode |
|---|---|---|---|---|---|
| M1 eager | 147 | | 8.4 | 87.2 | 50.2 |
| Bound, no CUDA Graph | 136.5 | | 8.0 | 80.4 | 47.2 |
| CUDA Graph | 106 | | 6.1 | 54.6 | 44.6 |
| + host DSP | 88 | | 6.2 | 54.6 | 26.8 |
| + context/fused step | **81.9** | **85.1** | 6.3 | 52.0 | 23.0 |
| PyTorch streaming reference | 199 | 265 | | | |

Real-time pacing (`--realtime --vad`, 22 s source): p50 90 ms, p95 107 ms, 102 blocks with zero overruns. About 8 ms higher under pacing than at full speed; that is the GPU downclocking while idle between blocks, which will not happen on a server with several streams running concurrently.

### Multi-stream (full speed, 12 s source, one round = every stream advances one block)

| Streams | Steps | Per-round p50 | Against the 240 ms period |
|---|---|---|---|
| 1 | 10 | 82 | Pass |
| 3 | 10 | 247 | Fail |
| 3 | 6 | 189 | Pass |
| 4 | 6 | 249 | Fail |

Rule: streams ≈ period × 0.8 / per-stream whole block. Per-stream cost does not change with the number of streams (DiT is compute-bound, M0 showed batching gives no gain), so this is the capacity of this card on this model.

### Quality

| Output | CAM++ cosine (against reference) |
|---|---|
| `mimic convert`, fused step | 0.661 |
| `mimic stream --vad`, 22 s | 0.669 |
| PyTorch streaming reference | 0.647 |
| Source against reference baseline | 0.27 / 0.12 |

The seven ORT parity tests (including the fused step at 10 steps) and the six dsp parity tests all pass, with the same numbers as M1.

## 4. What was not reached, and the headroom

- 10 steps p95 85 ms against the 80 ms gate. Breakdown: GPU graphs total 70.7 (XLSR 6.2, DiT 49.5, HiFT 14.7) + host DSP 7.7 + copies and solver overhead about 4. The only way through the gate is the GPU part: fp16 weights emitted directly to drop the Casts, ORT attention fusion, TensorRT EP (already listed in M0); the expected DiT single step 4.95 → 3.6 ms alone would save 13 ms. These are export-side work and do not change the Rust.
- Block length is a shape parameter of the graphs: changing it means re-exporting a set of graphs (`export_all.py --block`) into another directory. The CLI now gives a clear message on a shape mismatch.
- The noise gate is an energy gate, sufficient but simple; the trait is fixed and it can be swapped.
- Batching (batch N graphs) was not done: M0 showed no throughput gain, only scheduling simplification remains, not worth it for now.

## 5. Files

- `crates/mimic-onnx`: `BoundGraph`, `DeviceBuf` (cudart copies), `Graph` gains `set_input/run_set/run_set_keep/get_output/copy_output_to_input`
- `crates/mimic-core`: `Vad`, `FlowStepper`
- `crates/mimic-dsp`: `vad.rs`, rewritten `hift.rs`, `noise.rs` (Box–Muller pairs)
- `crates/mimic-model`: `DitStepFused`, `load_static` (goes through BoundGraph under CUDA and warms up)
- `crates/mimic-pipeline`: `Estimator` enum, `Session`/`Runner`, `solve_fused`
- `crates/mimic-cli`: `stream` subcommand, `--no-cuda-graph`
- `tools/export/export_all.py`: `DiTStepFused` export
