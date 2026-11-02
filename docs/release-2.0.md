English | [中文](release-2.0.zh-CN.md)

# Mimicking 2.0 official release (2026-11-01)

Mimicking 2.0 is the first version that runs real-time voice conversion **entirely on our own generator weights**: content encoding (XLS-R L12, Apache-2.0) → K=4096 soft-assignment content codebook → length regulator → DiT flow matching → HiFT vocoder (Apache-2.0); the regulator and the DiT are MimickingTrain's own definition with self-trained weights.
On the 32-pair LibriTTS-R evaluation, 2.0 reaches similarity to target 0.712 / to source 0.419 (English speaker model, streaming) and a streaming WER of 7.6 %; the real-time path is about 330 ms end to end at 160 ms blocks.

This document is the "frozen inventory + final parameters + numbers + known limitations". How to integrate is in `docs/integration-guide.md`; the pitfalls hit during training and diagnosis are in MimickingTrain `docs/lessons-learned.md`.

## 1. What the official release consists of

### Weights (MimickingTrain `runs/Mimicking-v2.0/`, not in git, archived together with the release)

| File | Content | sha256 (first 16 hex digits) |
|---|---|---|
| `ema.safetensors` | 2.0 generator EMA weights, one file with 108 tensors: 88 `dit.*` (DiT, 18.27 M parameters) + 20 `regulator.*` (regulator, 2.32 M), 20.6 M in total | `8d29af773a441995` |
| `xlsr12_k4096.safetensors` | content codebook: 4096 × 1024 centroids + tau (**required for inference**, baked into the graphs at export) | `f97ef495a7b56103` |
| `regulator_norm_stats.json` | whole-utterance statistics of the regulator's 4 GroupNorms (mean over 64 utterances, **required for export**) | `b70e30f31ad1f596` |
| `config.json` | fine-tune configuration (see §2) | `05f68988458a3b45` |
| `step_0010000.pt` | trainer checkpoint (model + optimiser + EMA), for resuming training | `7ac20d9fc4875ec1` |

### Exported model sets (`Mimicking/models/`, manifest `name` = `Mimicking-v2.0`, `version` = `2.0.1`)

| Directory | Block length | Right context | Purpose |
|---|---|---|---|
| `Mimicking-v2.0-lowlat` | 240 ms (configured 0.25 s, rounded to the 20 ms grid) | 0.02 s | **default set**, loaded when the Lab starts |
| `Mimicking-v2.0-b0.16-r0.02` | 160 ms | 0.02 s | latency first; GPU load about 19 % |
| `Mimicking-v2.0-b0.32-r0.02` | 320 ms | 0.02 s | compute first; 80 ms more latency |

Each set has 10 files: `xlsr.onnx`, `xlsr_ref.onnx`, `campplus.onnx`, `hift_*` (graphs of the Apache-2.0 models; exported by `mimictrain.export.shared` directly from the weights in `ref/models`) + our exported `length_regulator.onnx`, `length_regulator_ref.onnx` (fp32, codebook and frozen statistics baked in), `dit_step.onnx`, `dit_step_fused.onnx` (fp16 + ORT fused operators).
The manifest has two extra fields: `content_codebook` (codebook file, sha256, K, tau, which two graphs it is baked into) and `regulator_norm_stats` (source and values of the statistics). **Version 2.0.1 = 2.0 weights + frozen regulator statistics**; the 2.0.0 graphs smear in streaming (see §5 item 1) and have all been re-exported in place.

### Voice gate

`models/vad/silero_vad.onnx` (Silero VAD v6, MIT, 1.2 MB, sha256 first 16 hex digits `7ed98ddbad84ccac`, licence text in the same directory). `mimic_onnx::SileroVad` implements the `Vad` interface; about 1 ms per 240 ms block on the CPU.

## 2. Final parameters

### Model

| Stage | Parameters |
|---|---|
| Content encoding | `facebook/wav2vec2-xls-r-300m` layer 12, 1024-dim, 50 Hz, 16 kHz input, per-window mean/variance normalisation (inside the graph) |
| Content codebook | k-means K=4096 (`tools/content_gate.py fit`, XLS-R frames of the training set), soft assignment softmax(−d²/tau), tau = mean nearest squared distance of the training frames, tau_scale 1.0 |
| Length regulator | Linear 1024→384, nearest-neighbour interpolation to the 86.13 Hz mel frame rate, 4 × (Conv3 + GroupNorm(1,384) + Mish), 1×1 Conv; at inference GroupNorm uses frozen constants |
| DiT | 384-dim, 9 layers, 6 heads, RoPE, SwiGLU, U-ViT skip connections, time and timbre tokens; flow matching, 10 Euler steps + CFG 0.7 at inference |
| Style encoding | CAM++ `campplus_cn_common` (192-dim; 2.x plans to switch to an English model) |
| Vocoder | HiFT (CosyVoice-300M), 80-dim mel @ 22.05 kHz, hop 256 |

### Training (MimickingTrain)

| Stage | Settings |
|---|---|
| 1.0 (`runs/Mimicking-v1.0`, run name tiny_sp) | trained from scratch for 100 k steps; LibriTTS-R train-clean-100 / 360 / other-500; validation dev-clean |
| 2.0 fine-tune (`configs/ft_vq4096.json`) | warm-started from the 1.0 EMA, fine-tuned 10 000 steps after inserting the codebook; two A4000s (DDP, gloo), 24 000 frames per card per step; lr 5e-5, warmup 500, cosine to 0.2×; AdamW β (0.9, 0.98), wd 0.01, gradient clipping 1.0; EMA 0.9999; 3 s prompt, no-prompt probability 0.1; utterance length 90–1300 frames; input perturbation: pitch ±8 semitones (≥2), formant ±35 % (≥12 %), 3-band EQ ±12 dB, gain ±6 dB; 4 h 06 min |
| Freeze point | step 10 000 (similarity had plateaued after 2 500 steps; the attempt to continue to 30 k was stopped at step 11 400) |

### Export (`python -m mimictrain.export`)

`--block 0.25 --right 0.02` (also 0.16 / 0.32) `--prompt 3.0`, `--train-config` (bakes in the codebook) `--norm-stats` (frozen statistics) `--shared from-weights` (default: XLS-R / CAM++ / HiFT exported from the Apache weights, fixtures written to `fixtures/` inside the set) `--version 2.0.1`. Crossfade 0.05 s, left context 2.5 s (XLS-R) / 0.5 s (DiT) are the geometry defaults. Parity (`tools/export_parity.py`): regulator fp32 6.5e-4, CUDA TF32 2.7e-3; DiT fused graph 3.4e-4.

### Runtime (recommended values measured in the Lab)

| Item | Value | Notes |
|---|---|---|
| Block length | 160 or 240 ms | 29–30 ms of compute per block regardless of block length; GPU about 19 % at 160 ms, about 12 % at 240 ms |
| Right context | 0.02 s | the smallest grid step; raising it to 0.1 s does not help 2.0 |
| Denoising steps / CFG | 10 / 0.7 | |
| Gating | Silero VAD, threshold 0.5, closing needs 2 consecutive blocks below 0.35, absolute floor −60 dBFS | the energy gate is only the fallback when the model is missing; no gate for offline file conversion |
| Output margin | 50 ms (with locked GPU clocks) / 150 ms (unlocked) | locking: `nvidia-smi -i N -lgc 1800,1800`, p95−p50 drops from 60 ms to 2 ms |
| Queue control | playback-rate steering ±1 % (P: a 40 ms deviation is pulled back within 4 s; slow integral), pad with silence / drop input only as a backstop | absorbs the 0.1–0.3 % clock drift between microphone and speaker |
| Microphone level | about −25 dBFS while speaking; turn off the system's "audio enhancements / automatic gain" | −44 dBFS is too low, and AGC pushes the gain down for several seconds after a loud tap |
| Latency | 160 ms blocks: software path 234 ms (block 160 + compute 33 + margin 41) + algorithmic delay 80 ms + sound card 20–40 ms ≈ 330–350 ms; 240 ms blocks about +80 ms | algorithmic delay = right context 20 + crossfade 40 + SOLA search 20 |

## 3. Numbers

### Offline whole utterances (32 LibriTTS-R pairs, 3 s prompt; English CAM++ = VoxCeleb model, Chinese = campplus_cn_common)

| Version | English to target / to source | Chinese to target / to source | WER (source 6.3 %) |
|---|---|---|---|
| 1.0 | 0.635 / 0.524 | 0.842 / 0.735 | — |
| 2.0 (graphs 2.0.0) | 0.700 / 0.432 | 0.870 / 0.688 | 6.7 % |
| 2.0 (graphs 2.0.1, frozen statistics) | 0.708 / 0.439 | 0.874 / 0.690 | 7.2 % |

### Streaming (MimickingLab headless file mode, same 32 pairs, 240 ms blocks, 0.02 s right context, no gating)

| Set | WER | English to target / to source / above floor | Chinese to target / above floor |
|---|---|---|---|
| 2.0 graphs 2.0.0 | 16.1 % | 0.707 / 0.394 / +0.113 | 0.878 / +0.070 |
| **2.0 graphs 2.0.1 (official release)** | **7.6 %** | 0.712 / 0.419 / +0.138 | 0.875 / +0.069 |

4-pair comparison of 160 / 240 / 320 ms block lengths: similarity and WER show no difference within noise.

### Noise (BO's real recordings, number of open blocks at 240 ms blocks; without gating both model sets babble at full level)

| Recording | Energy gate | Silero gate |
|---|---|---|
| Keyboard (34.8 s) | 114 / 144 | 0 / 144 |
| Desk tapping (43.1 s) | 152 / 179 | 0 / 179 |
| Coughs and sneezes (15 s) | 44 / 62 | 7 / 62 |
| Office background with distant voices (34 s) | 133 / 141 | 40 / 141 (voices are let through by design) |
| Clean speech (recall) | 20 / 32 | 32 / 32 |

## 4. Weights and interface (for those replacing it)

- Weights: the regulator and the DiT are ours (1.0 trained from scratch + 2.0 codebook fine-tune); XLS-R, CAM++ and HiFT are the original Apache-2.0 models.
- Interface: the ONNX graph interface is fixed as XLS-R → regulator → DiT → HiFT, codebook and statistics are inside the graphs, and the Rust runtime loads by the directory's manifest.
- Quality: high similarity to the target with little leakage of the source speaker; streaming intelligibility close to offline; more sensitive to superimposed noise (§5).

## 5. Known limitations

1. **The regulator statistics of a codebook model must be frozen.** Graphs 2.0.0 used each segment's own GroupNorm statistics, streaming WER 16 %; 2.0.1 with frozen statistics 7.6 %. Every future export of a codebook model must carry `--norm-stats`, and the streaming WER must first be reproduced with `tools/stream_diag.py --segmented` before exporting.
2. **Tapping sounds mixed into speech eat words.** With keyboard noise 10 dB below the speech, 2.0.1's WER rises from 12 % to 31 %. Gating cannot handle noise during speech; the fix is a noise-augmented fine-tune, not done.
3. **The gate only separates speech / non-speech.** Other people's speech and some coughs and sneezes are let through; a speaker-similarity gate has not been built.
4. The style encoder is the Chinese CAM++; choosing references with it still works for English target timbres, but 2.x plans to switch to an English model.
5. Output is 22.05 kHz (11 kHz ceiling), determined by HiFT.
6. The prompt (reference timbre) takes the first 3 s of the reference audio; the CAM++ graph has dynamic shapes and does not use CUDA Graph, so changing voice takes about 350 ms.
7. Evaluation was done in English only (LibriTTS-R); French, accented speech and noisy scenes are not quantified.

## 6. Reproduction and verification

```bash
# Export (MimickingTrain, Mimicking's .venv; the whole set is self-contained; --shared from-weights is the default)
python -m mimictrain.export --weights runs/Mimicking-v2.0/ema.safetensors --train-config runs/Mimicking-v2.0/config.json \
    --norm-stats runs/Mimicking-v2.0/regulator_norm_stats.json --out ../Mimicking/models/Mimicking-v2.0-lowlat \
    --block 0.25 --right 0.02 --gpu 0 --version 2.0.1
# Parity ORT vs PyTorch
PYTHONPATH=. python tools/export_parity.py --dir ../Mimicking/models/Mimicking-v2.0-lowlat --weights runs/Mimicking-v2.0/ema.safetensors --train-config runs/Mimicking-v2.0/config.json --gpu 0
# File verification (Mimicking)
mimic models verify --models models/Mimicking-v2.0-lowlat
# Rust parity (fixtures in fixtures/ inside the set; the CUDA / cuDNN DLL directories must be on PATH)
MIMIC_ORT_DYLIB=<venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll MIMIC_GPU=1 MIMIC_MODELS=models/Mimicking-v2.0-lowlat cargo test --release -p mimic-model -p mimic-dsp -- --test-threads=1
# Reproduce the streaming measure (PyTorch segmented emulation matches the Lab: 2.0.1 7.2 %)
PYTHONPATH=. python tools/stream_diag.py --segmented --fixed-stats runs/Mimicking-v2.0/regulator_norm_stats.json --weights runs/Mimicking-v2.0/ema.safetensors --config runs/Mimicking-v2.0/config.json --gpu 0 --out runs/check --pairs 32
```

Data and records: MimickingTrain `docs/despeaker-design.md` §12–13 (fine-tune, freezing, Lab measurements, block length, right context, slurred-word diagnosis, noise), `runs/lab_32pair*/`, `runs/diag_*`, `runs/noisectx_*`.
