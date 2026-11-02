English | [中文](m1-status.zh-CN.md)

# M1 status: Rust core library + offline conversion chain (2026-10-26)

## Conclusion

**M1 acceptance passed.** Six crates compile; signal processing and the five graphs all pass stage-by-stage parity within tolerance; the audio produced end to end by `mimic convert` has the same speaker similarity as the PyTorch streaming version. Performance is not yet optimised (ORT eager, host-side copies, CUDA Graph not enabled); that belongs to M2.

## Code structure

```
Cargo.toml                 workspace (ndarray 0.17, ort 2.0.0-rc.15 with load-dynamic + api-22 against onnxruntime 1.22)
crates/
  mimic-core      traits: Graph, ContentEncoder, SpeakerEncoder, LengthRegulator, VelocityEstimator, Vocoder, MelAnalyzer, Resampler; Prompt
  mimic-dsp       SincResampler (torchaudio algorithm), MelSpec (seed-vc mel spectrogram), KaldiFbank (CAM++ input), hift (harmonic source, 16-point STFT/ISTFT), Sola, noise
  mimic-onnx      OnnxRuntime (loads the dylib at runtime, CUDA EP), OnnxGraph (implements Graph)
  mimic-model    WindowConfig/Shapes (window geometry, including numpy's round-half-to-even), XlsrEncoder, InterpolateRegulator, CampplusSpeaker, DitStep, HiftVocoder, ModelSet
  mimic-pipeline  EulerCfg solver, Stages, prepare_prompt, StreamSession (sliding window + SOLA), Converter (files go through the streaming session)
  mimic-cli       mimic convert --reference --source --out [--steps --cfg --block --extra-right --seed]
tools/export/     export_all.py (seven static graphs + fixtures), validate_ort.py, diag_placement.py, py_stream_convert.py (PyTorch streaming reference output), eval_outputs.py (CAM++ cosine)
```

The design points that differ from the original seed-vc are in place: single-step denoising graph + Rust Euler solver; the Prompt is an explicit object; the streaming state machine is independent of audio I/O and process boundaries; the core crates contain no knowledge of any integrating application.

## Stage-by-stage parity (Rust against PyTorch fixtures)

`cargo test -p mimic-dsp --release` (pure Rust, no GPU needed):

| Module | max abs | relative L2 |
|---|---|---|
| Resampling 22.05 k → 16 k (torchaudio sinc) | 4.2e-6 | |
| Mel spectrogram (22.05 k / 1024 / 256 / 80) | 2.1e-4 | 6.9e-7 |
| Kaldi fbank (mean-normalised) | 3.4e-4 | 3.7e-6 |
| HiFT 16-point STFT | 1.5e-7 | |
| HiFT ISTFT → waveform | 3.6e-4 | 5.0e-4 |
| Harmonic source RMS | 0.0282 vs 0.0282 | random phase, energy only |

`MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=1 cargo test -p mimic-model --release` (PATH must include torch/lib so that cuDNN is found):

| Stage | max abs | relative L2 | Notes |
|---|---|---|---|
| XLSR graph | 3.8e-2 | 3.9e-3 | fp16 |
| Length regulator graph | 6.0e-5 | 2.4e-4 | |
| CAM++ (Rust fbank → graph) | 3.6e-3 | 1.1e-3 | cosine 1.000000 |
| DiT single-step graph | 1.4e-2 | 8.8e-4 | |
| Euler 10 steps + CFG (Rust solver) | 1.7e-2 | 4.0e-4 | matches the Python ORT validation numbers |
| Full HiFT chain (same excitation source) | 2.6e-3 | 2.0e-3 | |

## End to end

Same source (`source_s1.wav`, 12.5 s), same reference (`azuma_0.wav`), GUI default geometry, 10 steps, CFG 0.7, GPU 1:

| | Rust `mimic convert` | PyTorch streaming reference (`py_stream_convert.py`) |
|---|---|---|
| CAM++ cosine (output vs reference) | **0.661** | 0.647 |
| CAM++ cosine (output vs source) | 0.691 | 0.690 |
| Source vs reference baseline | 0.266 | |
| Output level | −24.1 dBFS | −24.1 dBFS |
| Log mel distance between the two outputs | 0.66 (the same output shifted by one frame against itself gives 0.59, i.e. the difference is at the level of random noise) | |
| Per-block compute time (62 blocks) | 147 ms (content 8.4, regulate 1.2, solve 87.2, vocode 50.2) | median 199 ms, p95 265 ms |
| RTF | 0.73 | about 0.8 |

Output files: `out/rust_azuma_s1.wav`, `out/py_azuma_s1.wav`.

## Pitfalls hit during implementation

- Exporting with a dynamic batch axis puts the shape operators on the CPU and CUDA Graph capture is refused; fully static + onnxsim solves it. XLSR is exported once for the block window and once for the 3 s reference (`xlsr_ref.onnx`, weights duplicated, 329 MB; the weights can be shared later).
- Graphs exported under autocast output fp16; the export script uniformly appends a Cast at the end of the graph, and the Rust side only handles f32.
- `np.round` rounds half to even: a 0.25 s block = 12.5 units → 12; Rust must use `round_ties_even`, otherwise the sequence length is 502 instead of 499.
- On Windows the default integer in numpy 1.x is int32; npz scalar reads must accept i64/i32/f64.
- `GraphOptimizationLevel::Level3` from `ort` rc.15 is rejected by onnxruntime 1.22; the default is fine.
- rand 0.9 has no `StandardNormal`; a Box–Muller is included.

## Performance items for M2 (currently 147 ms → target ≤ 80 ms)

1. Device-resident I/O binding + CUDA Graph (in the ORT validation the DiT single step went 7.9 → 4.95 ms, whole-block GPU total 70.7 ms). `ort` has `IoBinding`, `Allocator`, `Tensor::new(device)`; what is missing is a host-to-device copy interface, which may need the CUDA driver API (cudarc) for memcpy, or the Euler update moved into the graph.
2. Host-side DSP: 9 Box–Muller draws per sample in the harmonic source and ndarray indexing are the bulk of the 50 ms vocode; pre-generate noise, remove bounds checks.
3. Batching/multi-stream, VAD gating, threading model.
