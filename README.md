English | [中文](README.zh-CN.md)

# Mimicking

A real-time zero-shot voice conversion library written in Rust: give it 16 kHz microphone blocks and it returns 22.05 kHz audio blocks in the new timbre, needing only a 3-second reference recording and no training for the target speaker.
The pipeline is XLS-R content encoding → content codebook + length regulator → DiT flow matching → HiFT vocoder; the generator (regulator + DiT) is the self-trained 2.0 weights from MimickingTrain. The models run as static ONNX graphs on ONNX Runtime's CUDA execution provider with CUDA Graph; everything around them is Rust.
On one RTX A4000 a 240 ms audio block takes about 30 ms, 4 streams per card; at 160 ms blocks the end-to-end latency is about 330 ms.

## Start here

| | Document | What it covers |
|---|---|---|
| Overview | [Project overview](https://infohex.github.io/Mimicking/overview.html) (source `docs/overview.html`) | Illustrated summary for decision-makers: what it does, how, latency and capacity, milestones |
| Flow | [Real-time pipeline flow](https://infohex.github.io/Mimicking/pipeline-flow.html) | How one 240 ms block becomes output: each stage's inputs, outputs, timing, code location, glossary |
| Tree | [Project tree](https://infohex.github.io/Mimicking/project-tree.html) | Every directory and file of the three workspaces, searchable |
| Release | [`docs/release-2.0.md`](docs/release-2.0.md) | 2.0 frozen inventory, final parameters, evaluation numbers, known limitations |
| Integration | [`docs/integration-guide.md`](docs/integration-guide.md) | How other projects integrate the library |
| Runtime | [`docs/runtime-pitfalls.md`](docs/runtime-pitfalls.md) | Hard rules for ORT / CUDA / threading |
| Licence | [`docs/license-review.md`](docs/license-review.md) | Item-by-item licence review; library code Apache-2.0 |

The three interactive pages show only their source when opened on GitHub; use the online addresses above (GitHub Pages) or clone the repository and open `docs/*.html` in a browser.

**2.0 official release (2026-11-01)**: default model set `models/Mimicking-v2.0-lowlat` (240 ms blocks; 160 / 320 ms sets also available); the voice gate is Silero VAD; on the 32-pair evaluation the English speaker model gives similarity to target 0.712 / to source 0.419 and a streaming WER of 7.6 %.
The frozen inventory, final parameters and numbers are in **`docs/release-2.0.md`**; how to integrate is in **`docs/integration-guide.md`**; the training and debugging pitfalls are in MimickingTrain `docs/lessons-learned.md`;
every directory and file of the three workspaces is described on the interactive page **`docs/project-tree.html`** (regenerate with `python tools/project_tree.py`).

## Structure

```
mimic-cli        command line: convert (file to file), stream (multi-stream real-time simulation), models (manifest verification)
mimic-pipeline   EulerCfg solver, Session/Runner streaming sessions, Converter; depends only on traits
mimic-core       stage traits: Graph, ContentEncoder, SpeakerEncoder, LengthRegulator, VelocityEstimator, FlowStepper, Vocoder, MelAnalyzer, Resampler, Vad
mimic-dsp        resampling (torchaudio algorithm), mel spectrogram, Kaldi fbank, HiFT harmonic source/STFT/ISTFT, SOLA, energy VAD
mimic-onnx       ONNX Runtime backend: dylib loaded at runtime, device-resident bindings, CUDA Graph
mimic-model     model package (the XLS-R → regulator → DiT → HiFT family): implementations of each trait, window geometry, manifest verification, into_stages()
examples/        convert-minimal: a complete conversion program in under 50 lines
tools/           package_runtime.py (assembles the runtime directory), project_tree.py (project tree page)
```

Dependencies point downwards: `mimic-core` and `mimic-pipeline` depend on no model or runtime; `mimic-model` is the first model package, and other model families can sit beside it.

## Quick start

```rust
use mimic_pipeline::{Converter, EulerCfg, Runner, Session};
use mimic_model::{Device, ModelFiles, OnnxRuntime, ModelSet};

let rt = OnnxRuntime::discover(None, Device::Cuda(0))?.with_cuda_graph(true);
let files = ModelFiles::new("models/Mimicking-v2.0-lowlat");          // 2.0 default set
let model = ModelSet::load(&rt, &files, files.config())?;              // geometry is read from the directory's manifest
let (stages, geo) = model.into_stages();
let mut runner = Runner::new(stages);
let prompt = runner.stages.prepare_prompt(&reference_22k, &geo)?;          // once per voice
let mut session = Session::new(geo, prompt, EulerCfg { steps: 10, cfg: 0.7 }, 1234);
let out_22k = Converter::convert(&mut runner, &mut session, &source_16k, |_, _| {})?;
// streaming: runner.push(&mut session, &block_16k)? -> Vec<f32> of block_frame samples at 22.05 kHz
```

The complete program is in `examples/convert-minimal/src/main.rs`. Command line:

```bash
mimic convert --reference ref.wav --source src.wav --out out.wav
```

```bash
mimic stream --reference ref.wav --source a.wav b.wav c.wav --realtime --vad --out-dir out/
```

```bash
mimic models verify
```

## Models and runtime libraries

Each subdirectory of `models/` is one set of static graphs + `manifest.json` (manifest `name` is `Mimicking-v2.0`, the codebook and the frozen regulator statistics are baked into the graphs, see `docs/release-2.0.md`); the runtime loads whichever the directory's manifest describes. The Silero voice gate lives in `models/vad/`.

The library does not bundle models. Sets are exported as a whole by MimickingTrain's `python -m mimictrain.export`: the regulator and DiT come from our weights, the XLS-R, CAM++ and HiFT
graphs are exported from the Apache-2.0 weights in `ref/models/` (XLS-R is loaded by its Hugging Face name `facebook/wav2vec2-xls-r-300m`), and the parity fixtures are written to `fixtures/` inside the set.
 Changing the block length / right context = re-exporting a set into a new directory with MimickingTrain.

Runtime library directory convention (`tools/package_runtime.py` can assemble it automatically from `.venv`):

```
<app>/
  mimic.exe
  runtime/onnxruntime/   onnxruntime.dll, onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll
  runtime/cuda/          cudart, cublas, cublasLt, cudnn*, cufft (CUDA 12 / cuDNN 9)
  models/Mimicking-v2.0-lowlat/   graphs + manifest.json (one directory per set)
  models/vad/            silero_vad.onnx (Silero VAD, MIT, the voice gate of `mimic_onnx::SileroVad`; without it only the energy gate is available)
```

The rules for multi-threading, multiple GPUs and reloading (one ORT environment per process, CUDA Graph capture must not run concurrently with other CUDA calls, one thread per GPU, and so on) are in `docs/runtime-pitfalls.md`; read it before changing the related code.

Lookup order: explicit `--ort` path → `$MIMIC_ORT_DIR` → `<exe>/runtime/onnxruntime` → `$ORT_DYLIB_PATH`; `<exe>/runtime/cuda` and `$MIMIC_CUDA_DIR` are prepended to `PATH`. The `ort` crate talks to onnxruntime 1.22 with `load-dynamic` + `api-22`.

## Tests

```bash
cargo test --workspace --release
```

The pure-Rust parity tests (resampling, mel spectrogram, fbank, HiFT STFT/ISTFT) read `MIMIC_FIXTURES`, otherwise `<MIMIC_MODELS or models/Mimicking-v2.0-lowlat>/fixtures` (mimictrain.export writes them inside the set); without fixtures they are skipped. The ONNX-side parity tests need a GPU:

```bash
MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=0 cargo test -p mimic-model --release -- --nocapture --test-threads=1
```

## Numbers (A4000, 10 steps, CFG 0.7)

2.0 official release (`Mimicking-v2.0-lowlat`, CUDA Graph, single stream; measured in MimickingLab): 29–30 ms per block regardless of block length (GPU load about 19 / 12 / 9 % for the 160 / 240 / 320 ms sets);
32-pair LibriTTS-R streaming WER 7.6 %, English CAM++ similarity to target 0.712 / to source 0.419; at 160 ms blocks, locked GPU clocks and a 50 ms output margin
the software path latency is 234 ms, plus the 80 ms algorithmic delay and the sound card buffers, about 330–350 ms end to end. The full tables are in `docs/release-2.0.md`.

The cost per stream does not change with the number of streams (the DiT is the compute bottleneck at this sequence length); the streams on one card run sequentially, and each stream's processing latency equals the whole round, so **at most 4 streams per A4000** (4 streams take 115 ms per round; 8 streams take 230 ms per round, right against the period, with doubled latency and no margin, unusable); two cards with 8 streams take about 120 ms per round per card. The complete concurrency capacity template (RTX A4000 16 GB, the same class as an RTX 3070; 1/2/4/6/8 streams on one card, 8/16 on two cards, power, VRAM, how to estimate for another card) is in `docs/capacity-a4000.md`. Stage-by-stage parity is all within fp16 noise. The records of the replication phase (M0–M3, export-side optimisation, the original plans) are in `docs/history/`.

## Licence

Code: Apache-2.0 (`LICENSE`, `NOTICE`). This repository contains only our own code and distributes no model weights or exported graphs. Models obtained separately keep their own licences: the 2.0 regulator and DiT weights are self-trained by MimickingTrain; XLS-R, HiFT and CAM++ are Apache-2.0; Silero VAD is MIT. The item-by-item review is in `docs/license-review.md`.
