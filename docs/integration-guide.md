English | [中文](integration-guide.zh-CN.md)

# Mimicking integration guide

For projects that want to use Mimicking for real-time voice conversion in their own product. Mimicking is a Rust library: you give it 16 kHz microphone blocks and it returns 22.05 kHz converted blocks; audio devices, queues, gating policy and configuration files stay in the integrating application's hands. This document follows the order of integration, and each section ends with the "things you must get right".

Versions and numbers are in `docs/release-2.0.md`; the background of the runtime rules is in `docs/runtime-pitfalls.md`.

## 1. Runtime environment

```
<app>/
  your.exe
  runtime/onnxruntime/   onnxruntime.dll, onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll (1.22)
  runtime/cuda/          cudart, cublas, cublasLt, cudnn*, cufft (CUDA 12 / cuDNN 9)
  models/Mimicking-v2.0-lowlat/        default set (240 ms blocks)
  models/Mimicking-v2.0-b0.16-r0.02/   160 ms blocks (optional)
  models/vad/silero_vad.onnx           voice gate
```

Lookup order: `OnnxRuntime::discover(explicit, device)` tries, in turn, the explicit path, `$MIMIC_ORT_DIR`, `<exe>/runtime/onnxruntime`, `$ORT_DYLIB_PATH`; `<exe>/runtime/cuda` and `$MIMIC_CUDA_DIR` are prepended to `PATH`. `tools/package_runtime.py` can assemble this directory from `.venv`. VRAM: one model set takes about 2.5 GB.

**Must get right**: a process has exactly one ONNX Runtime environment; the second call of `OnnxRuntime::load / discover` returns the same environment, and devices are switched with `with_device`. CUDA Graph capture (loading / warm-up) must not run concurrently with other threads' CUDA calls: load every card and every set first, then start real-time inference.

## 2. Choosing a model set

| Set | Block length | End-to-end latency (see §5) | GPU load (A4000, single stream) | When to use |
|---|---|---|---|---|
| `Mimicking-v2.0-b0.16-r0.02` | 160 ms | ≈ 330 ms | 19 % | latency first |
| `Mimicking-v2.0-lowlat` | 240 ms | ≈ 410 ms | 12 % | default, balance of capacity and latency |
| `Mimicking-v2.0-b0.32-r0.02` | 320 ms | ≈ 490 ms | 9 % | many concurrent streams first |

The three sets share the same weights and the same 0.02 s right context; sound quality and intelligibility show no difference within the evaluation noise. Switching block length within one family (same manifest `name`, same right context) is just reloading from a different directory. Verify files and sha256 with `mimic models verify --models <dir>`.

## 3. Minimal integration code

```rust
use mimic_pipeline::{EulerCfg, Runner, Session};
use mimic_model::{Device, ModelFiles, OnnxRuntime, ModelSet, SileroVad, CONTENT_SR, SR};

// once: runtime + model (geometry is read from the manifest)
let rt = OnnxRuntime::discover(None, Device::Cuda(0))?.with_cuda_graph(true);
let files = ModelFiles::new("models/Mimicking-v2.0-lowlat");
let model = ModelSet::load(&rt, &files, files.config())?;
let (stages, geo) = model.into_stages();
let mut runner = Runner::new(stages);

// once per voice: reference audio 22.05 kHz mono, the first 3 s are used
let prompt = runner.stages.prepare_prompt(&reference_22k, &geo)?;

// one session per call stream: gating + its own noise seed
let gate = SileroVad::load(&rt, std::path::Path::new("models/vad/silero_vad.onnx"))?;
let mut session = Session::new(geo, prompt, EulerCfg { steps: 10, cfg: 0.7 }, 1)
    .with_vad(Box::new(gate));

// real-time loop: exactly geo.block_16k 16 kHz samples in, geo.block_frame 22.05 kHz samples out, every time
loop {
    let block16: Vec<f32> = capture.next_block(geo.block_16k);      // your capture + resampling to 16 kHz
    let out22 = runner.push(&mut session, &block16)?;
    playback.push(&resample_22k_to_device(&out22));                 // your output resampling + queue
}
```

- `CONTENT_SR` = 16000, `SR` = 22050; `geo.block_16k` (160 ms block = 2560, 240 ms = 3840), `geo.block_frame` (number of 22.05 kHz output samples) and `geo.delay_samples()` (algorithmic delay, 80 ms) all come from the geometry; do not hard-code them.
- Changing voice: `session.set_prompt(new_prompt)`; window and SOLA state are kept, no seam.
- Offline whole file: `Converter::convert(&mut runner, &mut session, &source16k, |_, _| {})` goes through the same streaming path and aligns away the delay; do not attach a gate offline.
- Complete example: `examples/convert-minimal/src/main.rs`; command line `mimic convert / stream / models`.

**Must get right**: one card, one thread, one `Runner`; multiple streams = one Runner `push`ing several `Session`s in sequence (the per-stream cost does not change with the number of streams, one round = the sum of the streams, and it must be shorter than the block period); multiple cards = several threads, each with its own Runner. Slow work (CAM++ embedding, file IO, writing logs to disk) does not belong on the inference thread.

## 4. Audio pipeline

- **Input**: device sample rate → 16 kHz mono, accumulate a full `block_16k`, then `push`. Capture overruns must be dropped, not accumulated (a backlog is latency).
- **Output**: 22.05 kHz → device sample rate, into a ring queue consumed by the sound card callback.
- **Queue controller** (mandatory, otherwise latency climbs one way with every hiccup): before each block push, look at the queue depth's deviation from the target margin.
  - Clock drift (microphone and speaker have different crystals; USB microphones commonly differ by 0.1–0.3 %) is absorbed with **playback-rate steering**: error e (seconds) → rate correction c = clamp(e / 4 s + slow integral, ±1 %), and the output passes through one stage of variable-rate resampling (cubic interpolation is enough). The ear cannot hear 1 %, and the queue stops drifting.
  - Only real hiccups (queue below 1/4 of the margin or above margin + two blocks) are backstopped by padding with silence / dropping input. Every pad or drop is an audible edit; in normal use the count should be 0.
  - Do not use hard thresholds like "pad below half the margin, drop half a block above margin plus half a block" as the regular mechanism: at 0.25 % drift that cuts 80 ms every 30 s, which sounds like stuttering.
- **GPU jitter**: with a single stream the GPU downclocks between blocks and the per-block p95 is about 60 ms above p50. With locked clocks (`nvidia-smi -i N -lgc 1800,1800`, administrator) the difference is only 2 ms and the margin can drop from 150 ms to 50 ms. With several concurrent streams the GPU is never idle and the problem naturally disappears.
- **Microphone**: turn off the system's / driver's "audio enhancements, automatic gain, noise suppression" and set the speaking level to about −25 dBFS. Automatic gain pushes the gain down for several seconds after a burst of loud taps, which shows up as "cannot speak for a few seconds after typing".

## 5. Latency budget

| Stage | 160 ms block | 240 ms block | How to save |
|---|---|---|---|
| Accumulating one block | 160 | 240 | a shorter set (120 / 80 ms are also on the 20 ms grid, sound quality untested) |
| Compute per block (A4000, CUDA Graph) | 30 | 30 | independent of block length; steps 10 → 6 saves about 8 ms (sound quality untested) |
| Output margin | 50 (locked clocks) / 150 | 50 / 150 | locked clocks + rate steering |
| Algorithmic (right context 20 + crossfade 40 + SOLA search 20) | 80 | 80 | crossfade 0.05 → 0.02 s saves 30 ms (export parameter, seams untested) |
| Sound card in/out buffers | 20–40 | 20–40 | WASAPI exclusive / smaller buffers |
| **Total** | **≈ 330–350** | **≈ 410–430** | |

## 6. Gating

- Default `SileroVad`: recognises speech itself; keyboard, desk tapping and page turning do not open the gate, clean speech is fully recalled, onset delay 0–160 ms. Parameters: threshold 0.5 to open, 2 consecutive blocks below 0.35 to close, blocks below −60 dBFS are muted outright. About 1 ms of CPU per block; calling it on the inference thread is fine.
- `EnergyVad` only looks at loudness, and continuous typing opens it; it is only the fallback when the Silero model is missing.
- What the gate cannot separate: other people's speech (let through by design), some coughs and sneezes. When needed, add a speaker-similarity gate (CAM++ is already running) or a noise-augmented fine-tune; not done in 2.0.
- While the gate is closed the model keeps computing (the window has to roll); only the output fades to zero, so the moment the gate opens the model has the right context.
- Do not attach a gate to offline file conversion: a trimmed file starts with speech, the energy gate's floor gets pinned to the speech level, and the first seconds come out silent.

## 7. Multiple streams and capacity

The per-stream cost does not change with the number of streams (the DiT is the compute bottleneck at this sequence length); the streams on one card run sequentially and each stream's latency equals the whole round: **at most 4 streams per A4000** (240 ms blocks, about 115 ms per round), 8 streams on two cards. Details in `docs/capacity-a4000.md`. Load every stream before starting; a stream added while running must create its session in the gap between blocks (no CUDA Graph re-capture, a few milliseconds).

## 8. Acceptance

- File parity: `mimic convert --reference ref.wav --source src.wav --out out.wav --models models/Mimicking-v2.0-lowlat`; the output should match the Lab's file mode.
- Streaming metrics: MimickingLab `--headless --source S.wav --reference R.wav [--block 0.16] [--vad silero|energy|off]` prints per-block compute time percentiles, overruns, gate flips and CAM++ cosine; `--gate-probe FILE.wav` examines gating alone.
- Expected values (32 LibriTTS-R pairs, streaming at 240 ms): WER 7.6 %, English CAM++ similarity to target 0.71 / to source 0.42; 29–30 ms per block.
- Evaluation tools are in MimickingTrain: `tools/wer_eval.py`, `tools/pair_floor.py`, `tools/stream_diag.py`, `tools/noise_context.py`.

## 9. Common problems (symptom → cause → fix)

| Symptom | Cause | Fix |
|---|---|---|
| Latency over 2 seconds | an early set with a 2.0 s right context | use a `*-r0.02` set |
| Streaming articulation smeared, "pinyin-like", offline fine | the codebook model's graphs do not have the regulator statistics frozen (2.0.0) | use the 2.0.1 graphs; your own exports must carry `--norm-stats` |
| First seconds silent (file conversion) | energy gate attached offline | no gate offline |
| Model babbles when the keyboard is typed on | energy gate | Silero gate |
| A stutter every half minute | the queue controller drops input / pads silence to handle clock drift | playback-rate steering (§4) |
| Single stream: per-block time fluctuates, underruns | GPU downclocks between blocks | lock the clocks or increase the margin |
| CUDA error 901 when loading the second card | CUDA Graph capture concurrent with other CUDA calls | load sequentially (the library already locks; the integrator must not run inference while loading) |
| Changing the reference timbre stalls for 350 ms | the dynamic-shape CAM++ graph does not use CUDA Graph | compute `prepare_prompt` on a separate thread, then `set_prompt` when ready |
| Tapping sounds mixed into speech eat words | 2.0 is sensitive to superimposed noise (model limitation) | 2.x noise-augmented fine-tune; no runtime fix for now |

## 10. Licence

Library code Apache-2.0 (`LICENSE`, `NOTICE`). The repository distributes no model weights or exported graphs; models obtained separately keep their own licences: the 2.0 regulator and DiT weights are self-trained by MimickingTrain; XLS-R, HiFT and CAM++ are Apache-2.0; Silero VAD is MIT (`models/vad/LICENSE-silero-vad.txt`). Item by item in `docs/license-review.md`.
