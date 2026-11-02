English | [中文](training-plan.zh-CN.md)

# Self-trained weights: analysis of the training code and our own plan (proposal, 2026-10-27)

Goal: train the DiT flow-matching estimator and the length regulator from scratch with our own code, producing weights that we own; the training pipeline itself must not depend on
seed-vc or other GPL / non-commercial code; and it must be an order of magnitude faster than the original seed-vc training script. This document first analyses the provenance and problems of the original training path file by file,
then gives the plan; **no code is written without confirmation**.

## I. Composition and provenance of the original seed-vc training path

Training updates only two modules: `cfm` (the DiT estimator) and `length_regulator`. XLS-R, CAM++, HiFT and OpenVoice are frozen.

| File | Lines | Content | Provenance and licence | Disposition |
|---|---|---|---|---|
| `train.py` | 437 | training loop, feature extraction, timbre perturbation, loss | seed-vc original, GPL-3.0 | not used, rewrite |
| `data/ft_dataset.py` | 126 | directory scan, librosa loading, mel spectrogram, batch collation | seed-vc, GPL-3.0 | not used, rewrite |
| `optimizers.py` | 119 | multi-optimiser wrapper, AdamW, ExponentialLR | seed-vc, GPL-3.0 | not used, rewrite (a few dozen lines) |
| `modules/commons.py` | 476 | utility functions, `build_model`, `load_checkpoint` | mostly from VITS (MIT) | not used |
| `modules/diffusion_transformer.py` | 536 | DiT: Transformer/Attention/FFN/RMSNorm/RoPE from Meta gpt-fast (file header carries Meta copyright, BSD-3); the `TimestepEmbedder` comment points to OpenAI GLIDE (MIT); `modulate` and `FinalLayer` (adaLN) are verbatim identical to `models.py` in facebookresearch/DiT, **whose repository licence is CC BY-NC 4.0 (non-commercial)**, to be verified | mixed provenance, including a suspected non-commercial fragment | not used, rewrite from the architecture |
| `modules/flow_matching.py` | 167 | CFM loss and Euler sampling | structure and comments from Matcha-TTS / CosyVoice (MIT / Apache-2.0) | not used, write our own (about 60 lines) |
| `modules/length_regulator.py` | 141 | linear projection + nearest-neighbour interpolation + Conv/GroupNorm/Mish ×4 + 1×1 convolution | CosyVoice (Apache-2.0); imports the VQ from descript-audio-codec (MIT, unused in the tiny configuration) | not used, write our own (about 30 lines) |
| `modules/audio.py` | 82 | mel spectrogram | HiFi-GAN (MIT) | not used; the Rust `MelSpec` definition already exists, the Python side is rewritten against it |
| `modules/openvoice/*` | ~1000 | VITS converter used for timbre perturbation augmentation | OpenVoice V2 (MIT), weights MIT | not used, replaced with our own perturbation |
| `modules/campplus/*` | ~400 | CAM++ speaker encoder | 3D-Speaker (Apache-2.0) | not used, use our exported `campplus.onnx` |
| `modules/hifigan/*` | ~500 | HiFT vocoder | CosyVoice (Apache-2.0) | not needed for training; our exported ONNX for evaluation |
| transformers `Wav2Vec2Model` | — | XLS-R content encoder | Apache-2.0 | kept (the only third-party model code retained) |

Conclusion: on the training path, the only things worth keeping are XLS-R (via transformers) and the three Apache-2.0 weight sets. Everything else can be rewritten from the architecture, and
must be: seed-vc's DiT file is itself a splice of gpt-fast + GLIDE + facebookresearch/DiT, and the latter's CC BY-NC licence
contradicts the GPL repository licence; using it means inheriting an unclear licence chain.

## II. Problems with the original seed-vc training script (why it should not be used directly)

Walking through `train_one_step` (batch 2, up to 100 s of audio per batch):

1. **The frozen models are recomputed every step**: OpenVoice does one timbre conversion (including STFT and VITS inference), XLS-R runs twice (original + perturbed, 300M parameters, fp16),
   CAM++ computes fbank and embeddings one at a time in a Python loop. These take more than 90% of a step, while DiT itself has only 25M parameters.
2. **Data loading in the main process**: librosa decodes file by file (`num_workers=0` on Windows), the mel spectrogram is recomputed per utterance, no length bucketing, batches are zero-padded to the longest.
3. **DiT in fp32 throughout**, no autocast / bf16, no torch.compile; attention uses SDPA but the mask is expanded into a large `(B,1,T,T)` tensor.
4. **CFG dropout per whole batch** (`torch.rand(1)`), not per sample; timesteps sampled uniformly.
5. **No EMA**: the released checkpoint file name is `_ema`, which means the author's own training had EMA, but the public `train.py` does not; using this script as is
   cannot even reproduce the author's recipe.
6. No validation set, no evaluation, no warm-up, the learning rate is only ExponentialLR (×0.999996 per step), no gradient accumulation, v1 does not support multiple GPUs.
7. Loss and prompt masking loop over samples in Python.
8. Bloated dependency chain: munch, librosa, descript-audio-codec, modelscope, funasr, gradio and so on, none of which training needs.

Combined, these put a single A4000 at an estimated 0.5–1 s per step, more than half of it spent on the frozen models.

## III. Plan

### 3.1 Code location and dependencies

A standalone Python project `C:\Workspace\Rust\MimickingTrain` (for the same reason as MimickingLab: keep it out of the library repository; the model definitions also serve as the
new export path reused by Mimicking's `tools/export`). Dependencies limited to: torch, torchaudio (BSD), transformers (Apache, only for
XLS-R), safetensors (Apache), numpy, soundfile (BSD), onnxruntime-gpu (MIT, to run the CAM++ and HiFT ONNX during evaluation).
No seed-vc, OpenVoice, descript-audio-codec, librosa, munch, modelscope or funasr.

### 3.2 Model definitions (our own, about 400 lines)

Implemented from the architecture specification, not translated from the original files:

- **DiT**: 384 dimensions, 9 layers, 6 heads; RMSNorm + timestep-embedding modulation (adaLN form); RoPE; SwiGLU feed-forward (hidden dimension per gpt-fast's
  2/3·4d rounded to a multiple of 256, keeping the same shapes as the original weights); U-ViT skip connections (outputs of the first 4 layers fed to the last 4, linearly fused); time and timbre
  each take one token; input is [x, prompt_x, cond(, style)] concatenated and linearly projected; output MLP.
- **Length regulator**: Linear(1024→384) → nearest-neighbour interpolation to the acoustic frame rate → (Conv3 + GroupNorm + Mish)×4 → Conv1.
- **CFM**: linear path y = (1−(1−σ)t)·z + t·x₁, target u = x₁ − (1−σ)z, L1 loss, prompt region zeroed and excluded from the loss;
  Euler sampling and CFG exactly as on the Rust side.

**Acceptance**: write a key mapping that loads the original tiny weights into our definitions and compare layer by layer against the original forward pass (relative L2 ≤ 1e-5 in fp32).
Passing proves architectural equivalence; the ONNX exported afterwards still satisfies the Rust side's existing graph interface (x, prompt_x, t, style, mu → v; likewise the fused version),
with not a line of Rust changed. The parity is done on the development machine only; the original weights do not enter training.

### 3.3 Data pipeline

- Manifest building: scan directories, decode to 22.05 kHz mono, duration 1–30 s, loudness normalisation (−23 LUFS), optional VAD segmentation; write a parquet/jsonl manifest.
- **Offline cache of the small features**: mel spectrogram (same definition as the Rust `MelSpec`: 1024/256/80, slaney, log(max(x,1e-5))), CAM++ embedding
  (our `campplus.onnx`). About 60 MB per hour of audio, about 60 GB for 1000 h.
- **XLS-R computed online** (bf16, no_grad, batched after length bucketing, torch.compile): the 1024-dimensional 50 Hz features are 360 MB per hour; caching
  1000 h × (original + perturbed) would take more than 700 GB, not worth it; computing a batch of 160 s of audio online takes about 0.2–0.3 s, acceptable. Optional int8 cache for
  subsets used in repeated experiments.
- **Our own timbre perturbation**: the only purpose is to make the content features carry less timbre. Implement NANSY-style perturbation: formant shift (resampling + duration restoration), pitch shift,
  random parametric EQ, random gain, all done on the GPU with torch, with fresh random parameters every step. No OpenVoice, hence no need for `se_db.pt`.
  The prompt region uses the original features and the rest the perturbed features, as in the original seed-vc.

### 3.4 Trainer

- PyTorch 2.x, bf16 autocast, torch.compile, SDPA (flash), per-sample CFG dropout (10%), optional logit-normal timestep sampling.
- Length-bucketed dynamic batches (by frame budget, e.g. 20k frames per batch ≈ 4 minutes of audio), gradient accumulation, AdamW (β 0.9/0.98, wd 0.01) + linear warm-up +
  cosine decay, gradient clipping 1.0, **EMA** (0.9999).
- DDP: `torchrun` across the two A4000s (16 GB each is enough).
- Checkpoints and resumable training (safetensors + optimiser state), validation loss, every N steps synthesise fixed test utterances with the HiFT ONNX and compute CAM++ cosine,
  TensorBoard logging.
- Estimate: with the cache ready, about 0.3–0.4 s per step on one card (about half of it XLS-R online), two-card DDP doubles throughput; 300 k steps in about 1–2 days.

### 3.5 Export and evaluation

- Export: move the fused-operator symbolic code from `export_dit_opt.py` onto our model definitions, producing ONNX files and a manifest with the same names;
  from then on the export script no longer imports seed-vc.
- Evaluation: fixed test set (CAM++ cosine and Whisper WER on the same basis as the existing `eval_outputs.py`) compared against the original tiny; the test bench runs real-time and 4-stream load.

### 3.6 Phases

| Phase | Content | Output | Estimate |
|---|---|---|---|
| T0 | model definitions + parity against the original weights + new export path | proof of architectural equivalence; tools/export free of GPL dependencies | 2–3 days |
| T1 | data pipeline + perturbation + feature cache | manifest and cache for 1 commercially usable dataset | 3–5 days (including download) |
| T2 | trainer + single-card smoke test (overfit a small dataset) | evidence that training converges | 2–3 days |
| T3 | full training (two cards) + evaluation + export | first version of our own weights + evaluation report | training 1–2 days, evaluation 1–2 days |

Data prerequisite: a commercially usable speech corpus. English: LibriTTS-R (CC BY 4.0), MLS (CC BY 4.0); Chinese: AISHELL-3 (Apache-2.0) is too small,
separate licensing or our own collection would be needed. This must be decided before T1.

## V. Progress

- **T0 complete (2026-10-27)**: `C:\Workspace\Rust\MimickingTrain`. Our own DiT and length regulator pass bit-level/1e-7 parity against the original modules;
  the export path `python -m mimictrain.export` does not import seed-vc, and the graphs it produces verify in ORT and on the Rust side (`mimic convert`),
  with the same timbre cosine as the old export. Finding: of the 25.68 M parameters in the original cfm, 7.4 M are dead modules unused in the forward pass; effective parameters 18.27 M.
  BO has decided: code goes in MimickingTrain, first version on public English data. Next: T1 (data pipeline).

- **T1 complete (2026-10-27)**: downloader (resumable, md5, EU mirror 33 MB/s), manifest (1–30 s filter), mel spectrogram with the same definition as Rust/the original
  (< 1e-5 against the original), CAM++ fbank, sharded safetensors cache (int16 waveform + fp16 mel spectrogram + fp16 timbre vector), our own timbre perturbation
  (pitch ±1.5–6 semitones, formants ±8–25%, random EQ, gain; median CAM++ cosine between original and perturbed about 0.35).
  All seven LibriTTS-R subsets cached and checked item by item (cache 119 GB, `C:\MimickingData`; the archives were once on a USB hard disk, and after the cache was held back by seeking everything
  was moved to the SSD, see the MimickingTrain README for details). Next: T2 (trainer).

| Subset | Items | Hours | Speakers |
|---|---|---|---|
| train-clean-100 | 31717 | 53.1 | 247 |
| train-clean-360 | 110924 | 188.9 | 899 |
| train-other-500 | 194851 | 306.0 | 1160 |
| **Training total** | **337492** | **548.0** | **2306** |
| dev-clean / dev-other | 5490 / 4353 | 8.9 / 6.3 | 40 / 33 |
| test-clean / test-other | 4615 / 4679 | 8.5 / 6.6 | 38 / 33 |
| All | 356629 | 578.3 | — |

- **T2 complete (2026-10-27)**: trainer `mimictrain/train/` (bf16, length-bucketed frame-budget batches, per-sample CFG dropout, EMA, two-card DDP (gloo),
  online XLS-R run twice for original + perturbed, checkpoints and resumption, validation loss, HiFT synthesis + CAM++ similarity evaluation). The length regulator now interpolates per sample length, with
  masked GroupNorm; batched training and single-utterance inference are bit-identical. Perturbation reimplemented fully batched (2.5 s → 37 ms per batch). Measured about 400× real time on two cards,
  one epoch (548 h) about 1.3 h, the default 100 k steps about 1.5 days. Baseline: original weights validation loss 0.606, CAM++ out↔ref 0.848. Next: T3 (official training).

### Measured compute and training-time estimate (rough measurement on the morning of 2026-10-27, single A4000; for the accurate numbers after T2 see the previous entry and the MimickingTrain README)

| Stage | Measured | Equivalent |
|---|---|---|
| DiT + length regulator training step (bf16, not compiled) | 16 × 10 s = 160 s of audio / 536 ms | about 300× real time |
| XLS-R 12-layer forward (bf16, frozen), run twice per step (original + perturbed) | 16 × 10 s / 280 ms | about 580× real time, about 290× after two passes |
| Total per card | | about 140× real time |
| Two-card DDP | | about 270× real time |

LibriTTS-R training subsets, 548 h: one epoch about 2.0 h. At 20 epochs (enough for a first listenable result) about 2 days; at 40–50 epochs (about 300 k steps,
320 s of audio per step) about 4–5 days. The original seed-vc script runs at about 40–80× real time on one card with no multi-GPU; ours is about 3–7× faster. Further speed-ups: torch.compile,
caching XLS-R features for the original audio only (saving one forward pass), larger batches; together another 1.5–2× is plausible.

## IV. Decisions to confirm

1. Whether the plan as a whole is approved; whether putting the code in the standalone directory `MimickingTrain` is agreed.
2. Whether the first version does English first (data available) or goes straight to Chinese (data to be solved first).
3. T0 parity may load the original weights on the development machine for verification (verification only, no training, no distribution).
