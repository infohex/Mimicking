English | [中文](rust-realtime-vc-plan.zh-CN.md)

# Mimicking: plan for an independent Rust re-implementation and optimisation of seed-vc (v3)

Status: proposal v3, not started.
- v1: standalone desktop tool. v2: built directly as a replacement worker for RPIVoiceGW.
- **v3 (decided by BO 2026-10-26)**: Mimicking is a standalone project; first replicate and optimise seed-vc, with a modular design whose boundaries we define ourselves, not a line-by-line copy; once the port succeeds it is reused by other projects in a modular way. RPIVoiceGW is the first candidate integrating application, but it is out of scope for this phase.

Basis: measurements in `bench/RESULTS.md`.

## 0. In one sentence

A Rust workspace whose core is a set of **library crates with boundaries defined by traits**: content encoding, timbre encoding, length regulation, flow-matching denoising, vocoder, resampling and VAD are each a replaceable module, composed by one pipeline object into an offline interface "reference audio + source audio → converted audio" and a streaming interface "push a block in, get a block out". The first implementation set corresponds to the seed-vc tiny weights; the models run as ONNX static graphs on the ONNX Runtime CUDA execution provider with CUDA Graph. Executables are only thin shells: this phase builds just one command-line tool for validation and measurement; adapter layers for the gateway or other projects are decided after the port succeeds.

## 1. Principles

| Principle | Meaning |
|---|---|
| Library first | Every capability lives in a crate; binaries only do argument parsing and I/O. Every integrating application (command line, gateway worker, other services later) is a thin adapter layer |
| Boundaries are ours | Modules are cut by algorithm stage, not by the original seed-vc Python files. The original `real-time-gui.py` kneads window management, model calls and SOLA into one callback; here they are separated |
| Weight-compatible, code not copied | Must be able to load the weights published by the original seed-vc (otherwise no parity, no zero-shot); but the solver, caching, streaming state machine and batching are our own design |
| Swappable backend | The inference backend is a trait; the first set uses ORT; later TensorRT, DirectML, even other model families leave the upper layers unchanged |
| Everything testable | Every module has parity fixtures; the pipeline has end-to-end metrics; streaming has per-segment timing |
| No assumed integrator | The core crates do not know what 8 kHz telephony is, what a pipe protocol is, or what config.ini is |

## 2. Constraints given by the measurements (unchanged)

| Measurement | Requirement |
|---|---|
| DiT single forward 14.2 ms, CPU launch of 2619 operators 13.95 ms, CUDA Graph replay 3.6 ms | Execute the model as a whole graph, not operator by operator |
| Time unchanged from 128 to 1024 frames; batch 1 and 2 identical (PyTorch eager) | Fixed shapes. **M0 correction**: this was an illusion of scheduling overhead hiding the real compute; with graph execution DiT time grows linearly with batch (about 5 ms per stream per step), batching gives no throughput gain, see `m0-results.md` §3 |
| `.max()` in `sequence_mask()` synchronises every step | Removed at export |
| XLSR and HiFT suffer from the same scheduling drag | All through static graphs |
| Pure-CPU periphery 7 ms | The periphery is not the focus, but it must be on a separate thread from GPU submission |

## 3. Architecture

### 3.1 Crates and dependency direction

```
mimic-cli (binary, for validation and measurement)
   │
mimic-pipeline      offline conversion + streaming session + batching; depends only on the traits below
   │
mimic-core          traits and data types: ContentEncoder, SpeakerEncoder, LengthRegulator,
   │                FlowDenoiser, Vocoder, Resampler, Vad, Backend; types such as Audio, Mel, Features
   ├── mimic-dsp         pure Rust signal processing: resampling, mel spectrogram, Kaldi fbank, ISTFT, SOLA, fades, energy VAD
   ├── mimic-onnx        ORT implementation of Backend: sessions, fixed-shape buffers, CUDA Graph, device selection, timing
   └── mimic-model      the seed-vc tiny "model package": implementations of each trait (composing ORT graphs + dsp), shape computation,
                         configuration (block length, context, prompt length, steps, CFG), weight manifest and verification
tools/export/ (Python, in .venv)  export ONNX from ref/seed-vc, generate parity fixtures
```

Rules: `mimic-core` and `mimic-pipeline` depend neither on ORT nor on any model; `mimic-model` is the first model package, and other model families later are sibling crates.

### 3.2 Core traits (intent, not final signatures)

| trait | Input → output | seed-vc tiny implementation |
|---|---|---|
| `ContentEncoder` | 16 kHz waveform window → 50 Hz content features | first 12 layers of XLSR-300m (ORT) |
| `SpeakerEncoder` | 16 kHz reference waveform → timbre vector | Kaldi fbank (dsp) + CAM++ (ORT) |
| `LengthRegulator` | content features + target frame count → conditioning sequence | nearest-neighbour interpolation + convolution (ORT) |
| `FlowDenoiser` | conditioning + prompt mel + timbre + noise + steps/CFG → mel | single-step DiT graph (ORT) + Rust solver (Euler; swappable) |
| `Vocoder` | mel → waveform | HiFT body (ORT) + ISTFT (dsp) |
| `Resampler` / `Vad` | waveform → waveform / gating | dsp |
| `Backend` | load graph, bind fixed-shape buffers, run, whether graph capture is enabled | ORT CUDA EP |

The two objects above `mimic-pipeline`:
- `Converter`: offline. `prepare_reference(wav) -> Prompt` (cacheable, serialisable); `convert(prompt, wav) -> wav`.
- `StreamSession`: streaming. Constructed from block length, context and prompt length, `push(block_16k) -> Option<block_out>`; internally manages the rolling window, SOLA, VAD gating and per-segment timing. `BatchRunner` merges the same period of several `StreamSession`s into one graph run.

The output sample rate is the model's 22.05 kHz; downsampling to 8 kHz, cutting 20 ms frames and the like belong to the integrating application; `mimic-dsp` provides a resampler but the pipeline makes no decisions on the integrator's behalf.

### 3.3 Design points that differ from the original seed-vc

- Single-step denoising graph + Rust solver: steps, CFG and solver are all variable on the Rust side; the graph is unchanged.
- `Prompt` is an explicit object rather than a global variable; it can be cached by key and saved to disk for reuse.
- The streaming state machine is independent of audio I/O and process boundaries, with a purely functional interface, convenient for testing and batching.
- Batching is kept as a scheduling device (one graph run serves several streams, fewer submissions), not a throughput device: M0 measured DiT at about 5 ms per stream per step, growing linearly with batch. Fixed `max_batch`, zero-padded when not full.
- Masks, positional encoding and `freqs_cis` are baked into constants per shape at export; there are no synchronisation points at runtime.
- VAD is a trait, with an energy-threshold implementation by default; the integrator can swap it.

### 3.4 Model export (`tools/export/`)

| Module | Export method | Fixed shapes (block 0.25 s as the example) |
|---|---|---|
| XLSR first 12 layers fp16 | HF export, then truncate layers | N×76800, output N×239×1024 |
| Length regulator | export, target length constant | N×T_cond×384 |
| DiT single step | export the single step only; CFG's batch 2 goes into the graph; constant masks | 2N×80×(258 prompt frames + T_cond + 2) |
| CAM++ | export, dynamic length | 1×T×80 |
| HiFT | body up to magnitude + phase spectrum; ISTFT in Rust; if the harmonic source cannot be exported it also moves to Rust | N×80×T_mel |

The export scripts only read `ref/seed-vc` and its checkpoints; the necessary changes (removing `.max()`, constant masks) are written in the export script's own copy of the model, without modifying `ref/`.

### 3.5 Runtime choice

| Option | Verdict |
|---|---|
| **ONNX Runtime (`ort`) CUDA EP + CUDA Graph** | First choice: graph execution solves the measured bottleneck; fp16; the TensorRT EP and DirectML EP share the same code |
| candle | No graph capture; Rust operator-by-operator launch at 3 to 5 ms is of the same order as the GPU time; dropped |
| burn | The Windows CUDA backend is too new; wait and see, the `Backend` trait leaves the door open |
| tch-rs | The same dispatcher called from another language, does not solve the problem; dropped |

### 3.6 Configuration

One typed `WindowConfig` (block length, CE context, DiT context, right context, prompt length, steps, CFG, max batch, device). Block length, context, prompt length and max batch determine the graph shapes and are construction-time parameters; steps and CFG can be changed at runtime. Defaults match the original seed-vc GUI (0.25 s / 2.5 / 0.5 / 2.0 / 3 s / 10 / 0.7) to allow parity.

## 4. Validation

### 4.1 Stage-by-stage parity (fixtures produced by `tools/export/`, consumed by Rust tests)
Resampling, mel spectrogram, fbank relative error ≤ 1e-3; each ONNX graph against PyTorch fp16 max absolute error ≤ 1e-2; the whole chain with the same noise seed, output mel spectrogram L1 ≤ threshold.

### 4.2 End-to-end metrics
On a fixed set of source utterances and references, compare the Rust and Python outputs with the same objective metrics: speaker similarity (Resemblyzer or CAM++ cosine), WER (Whisper transcription). The difference must be within noise.

### 4.3 Performance acceptance gates (A4000, block 0.25 s, 10 steps, CFG 0.7, single stream)

| Metric | Current (Python) | Target | M0 measured in ORT (GPU graphs total) |
|---|---|---|---|
| Single-stream per-block time p95, 10 steps | 258 ms (with VAD) | ≤ 80 ms | 70.7 ms + host DSP |
| Single-stream per-block time, 6 steps | about 170 ms | ≤ 60 ms | 50.9 ms |
| DiT 10 steps | 155 ms | ≤ 55 ms | 52 to 56 ms (including host-side Euler update) |
| Multi-stream | not supported | per-stream cost no more than single stream × 1.1; streams = block length × 0.6 / per-stream whole block | DiT linear: 5 ms per stream per step |
| Process start to ready for inference | 5 to 12 s | ≤ 5 s | not measured (Python ORT session creation takes a few seconds) |

## 5. Milestones

### M0: export and runtime validation (Python) — completed 2026-10-26, see `m0-results.md`
Export five graphs; run the whole block in Python onnxruntime-gpu with CUDA EP + CUDA Graph; measure single stream and batch 2/4/8/16; parity against PyTorch.
**Result**: DiT single step 4.95 ms (gate ≤ 5); whole block 10 steps 70.7 ms (gate ≤ 80); accuracy all relative L2 ≤ 4e-3; the "batch 4 ≤ 120 ms" gate voided because its premise was wrong, replaced by per-stream cost.
Remaining optimisation items (after M1): fp16 weights emitted directly to drop the Casts, ORT attention fusion, TensorRT EP (runtime libraries not installed on this machine).

### M1: core library + offline conversion — completed 2026-10-26, see `m1-status.md`
`mimic-core`, `mimic-dsp`, `mimic-onnx`, `mimic-model`, `mimic-pipeline` (solver, streaming session, Converter), `mimic-cli convert`.
**Result**: the six dsp and six ORT stage-by-stage parity tests all pass; end-to-end CAM++ cosine 0.661 (PyTorch streaming version 0.647); 147 ms per block (eager, CUDA Graph not enabled) against PyTorch 199 ms.

### M2: streaming session and performance — completed 2026-10-26, see `m2-status.md`
Device-resident I/O binding + CUDA Graph, faster host DSP, fused Euler step, `Runner`/`Session` multi-stream, energy VAD, `mimic stream`.
**Result**: single stream 82 ms p50 / 85 ms p95 per block; single stream under real-time pacing with zero overruns; batching not done, per the M0 conclusion.
**Export-side GPU optimisation (same day, see `gpu-opt.md`)**: DiT forward rewritten to emit ORT fused operators, whole graph fp16, single step 4.95 → 2.10 ms; HiFT body fp16 with overflow-safe Snake 14.3 → 10.2 ms. Single stream **48 ms p50 / 53 ms p95** (gate 80 passed); on the A4000 4 streams at 10 steps, 5 streams at 6 steps.

### M3: release as a library — completed 2026-10-26, see `m3-status.md`
README, rustdoc with zero warnings, `examples/convert-minimal` (40 lines), `manifest.json` + `mimic models verify`, `OnnxRuntime::discover` runtime library directory convention, `tools/package_runtime.py` (3.0 GB self-contained directory, verified in a clean environment), GitHub Actions CI (Windows + Ubuntu, manual GPU job), CHANGELOG 0.1.0, first git commit.
**Acceptance gate**: passed. Pending: finalising the code licence and the LICENSE file.

### M4: integrator adaptation (decided separately after the port succeeds)
For example a worker protocol adapter for RPIVoiceGW: a thin shell calling `StreamSession` and `BatchRunner`, doing 22.05 k → 8 k and protocol framing. Which repository it goes in and which configuration changes are decided then.

### M5: model-level acceleration (optional, requires training or re-export)
Step distillation, CFG distillation, TensorRT int8, KV cache for the prompt segment.

## 6. Risks

| Risk | Mitigation |
|---|---|
| HiFT's ISTFT and harmonic source cannot be exported | Already planned for: Rust implementation |
| ORT's CUDA Graph silently falls back on some operator | Measure it as the very first thing in M0, confirm with ORT profiling |
| XLSR's real GPU time in ORT is high | Measured in M0 too; TensorRT EP as the alternative |
| Premature trait abstraction costs performance (cross-module copies) | Zero-copy slices + preallocated buffers for the data types; benchmarks hold the line of §4.3 |
| GPL-3.0 and the weight licence | Rewrite from the paper and the weights, no original seed-vc code; check the weight licence on Hugging Face before starting |
| The original seed-vc repository is archived | Export depends only on the `ref/` snapshot and fixed weights |

## 7. Decisions pending

1. M0 uses GPU 1 (GPU 0 is kept for other experiments), as the default.
2. Directories: `crates/` for the crates above, `tools/export/` for Python, `models/` for the exported ONNX (not checked in), `bench/`, `docs/`, `ref/` stay as they are.
3. Whether to start M0 now.
