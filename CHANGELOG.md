English | [中文](CHANGELOG.zh-CN.md)

# Changelog

## 2.0 — 2026-11-01

**Mimicking 2.0 official release.** The generator is now our self-trained weights (`Mimicking-v2.0`, graph version 2.0.1), the default set is `models/Mimicking-v2.0-lowlat`, with further
sets for 160 / 320 ms blocks (the 0.1 s right-context experiment set has been deleted; its finding, no benefit for 2.0, is recorded in release-2.0); Silero VAD voice gate; MimickingLab's queue controller now absorbs clock drift with playback-rate steering.
Offline and streaming evaluation, final parameters and known limitations are in `docs/release-2.0.md`; how to integrate is in `docs/integration-guide.md`. The entries included in this version follow.

- Docs (2026-11-01): the formal documents (README, release-2.0, integration guide, flow page) describe 2.0 on its own terms, without
  the baseline comparison; the replication-phase records (M0–M3, gpu-opt, the original plans) moved to `docs/history/`.
- Housekeeping (2026-11-01): the manifest block the runtime reads is `provenance` (alias `upstream` for the baseline sets);
  `mimic` and the parity tests default to the first set with a manifest under `models/` (`*-lowlat` preferred) instead of the
  removed `models/onnx`; `tools/package_runtime.py --set` packages one named set; workspace version 2.0.0.
- Licence (2026-11-01): the library code is Apache-2.0 (`LICENSE`, `NOTICE`; Cargo `license = "Apache-2.0"`). Weights keep their
  own licences and the repository still distributes none.
- Rename (2026-11-01): the model-package crate `mimic-seedvc` is now `mimic-model`, `SeedVcConfig` → `WindowConfig`,
  `SeedVcTiny` → `ModelSet`, `MelSpec::seedvc_22k()` → `mel_22k()`. The code was always our own; only the names dated from the
  M1–M3 replication. CLI, example, MimickingLab, CI and docs updated; the seed-vc name now appears only as provenance.
- Self-contained export: the 2.0 sets' `xlsr*`, `campplus` and `hift_*` graphs are now exported by MimickingTrain
  (`mimictrain.export.shared`) from the Apache-2.0 weights in `ref/models/` (hift.pt, campplus_cn_common.bin; XLS-R by
  Hugging Face name), so no GPL code is involved anywhere in the 2.0 toolchain; the manifest's `upstream` block became
  `provenance`. Every graph's I/O is checked to be fp32. The Rust parity tests take `MIMIC_MODELS` / `MIMIC_FIXTURES`,
  read the geometry from the manifest and find fixtures next to the graphs (`<set>/fixtures`); the regulator bound is
  5e-3 (TF32 + codebook). `tools/export`, `ref/seed-vc` and `bench/` (baseline-only material, GPL-adjacent) were
  archived to `C:/Workspace/Rust/_archive/Mimicking-seedvc-2026-11-01`; the exported baseline sets `models/onnx*` stay for comparison.

- `mimic-onnx::SileroVad`: Silero VAD (v6 `silero_vad_16k_op15.onnx`, MIT, `models/vad/silero_vad.onnx` + license) behind the
  `Vad` trait, CPU session on the shared ORT environment, 64-sample context + 512-sample chunks, block decision with
  the energy gate's open/hold conventions and absolute floor. On real recordings the energy gate opened on 75–95 % of
  keyboard / desk-tap blocks (loudness alone cannot tell typing from speech) and the converter then babbles at full
  level; Silero keeps them at 0 blocks with full speech recall. `mimic-cli --vad` still means the energy gate.

- 2.0 sets re-exported as 2.0.1 with the length regulator's GroupNorm statistics frozen to dataset constants
  (manifest `regulator_norm_stats`; no runtime change). With the content codebook the per-block statistics swung enough to
  garble streaming output (32-pair streaming WER 16.1 % vs baseline 7.9 %; offline both ~6.5 %); frozen, the streaming
  emulation gives 7.2 %. Diagnosis in MimickingTrain `docs/despeaker-design.md` §13.5, tool `tools/stream_diag.py`.

- `models/Mimicking-v2.0-b0.16-r0.02` and `models/Mimicking-v2.0-b0.32-r0.02`: the 2.0 generator exported for 160 ms and
  320 ms blocks (same weights, same 0.02 s right context as `Mimicking-v2.0-lowlat`; `onnx-b0.32-r0.02` is the matching
  baseline set). MimickingLab switches between them with its block-length selector.

- Model sets trained by MimickingTrain are named `Mimicking-v<major.minor>` (directory `models/Mimicking-v2.0-lowlat`,
  manifest `name`), so they are told apart from the baseline seed-vc sets (`onnx`, `onnx-lowlat`, …) at a glance.

- Generator graphs can now be exported by `MimickingTrain` (`python -m mimictrain.export`), a clean-room
  definition of the DiT and length regulator with no seed-vc import; parity against the original seed-vc module is
  3e-7 relative L2 and the graphs load in the Rust runtime unchanged (`models/onnx-t0-lowlat`).

- `mimic-onnx`: `OnnxRuntime::load` returns the process's existing environment on every call after the first
  (ORT allows one per process); `BoundGraph::load` / `warm_up` hold a process-wide lock so CUDA Graph capture
  never overlaps another thread's CUDA calls (concurrent loads failed with CUDA error 901). Rules collected in
  `docs/runtime-pitfalls.md`.

- `mimic-onnx`: `OnnxRuntime::with_device` / `device()`. ORT allows one environment per process, so a host
  keeps its runtime and switches device on it (GPU → CPU fallback, reloading models) rather than loading again.

- `mimic-pipeline`: the VAD decision is applied to the output block it belongs to (queued by the
  algorithmic delay in blocks: 9 at a 2 s right context, 0 at 0.02 s). Gating with the current decision cut
  speech wherever "now" paused while the audio being played did not. With lookahead available the gate
  pre-opens one block, and the transition ramp is the cross-fade length (~40 ms) instead of a whole block.
- `mimic-dsp`: `EnergyVad` opens only on sustained energy (3 of 4 block quarters above threshold), holds two
  blocks before closing, absolute floor -60 dBFS (was -48, which muted quiet microphones outright).

- `mimic-onnx`: ORT thread pools no longer spin (`session.intra_op/inter_op.allow_spinning = 0`); seven
  sessions' worth of spinning threads is wasted CPU next to a host's audio threads.
- `mimic-model`: `WindowConfig::from_manifest` and `ModelFiles::config()` read the window configuration from
  `manifest.json`; `mimic` loads a model directory with its own geometry, `--block` / `--extra-right` are
  overrides only.
- `tools/export`: `MIMIC_BLOCK`, `MIMIC_EXTRA_RIGHT`, `MIMIC_ONNX_DIR`, `MIMIC_FIX_DIR` let several window configurations
  coexist; a low-latency set (right context 0.02 s, the shipped default of the original seed-vc GUI) exports in about two
  minutes. The 2.0 s right context of the default set was the original seed-vc dataclass default, not what the GUI
  ships, and is the whole reason for the 2.3 s algorithmic delay.

## 0.1.0 — 2026-10-26

First usable release of the library and the `mimic` CLI.

- Crates: `mimic-core` (stage traits), `mimic-dsp` (resampler, mel, Kaldi fbank, HiFT source/STFT/ISTFT, SOLA,
  energy VAD), `mimic-onnx` (ONNX Runtime backend, device-resident bound graphs, CUDA Graph), `mimic-model`
  (seed-vc tiny model package, manifest verification), `mimic-pipeline` (Euler/CFG solver, streaming sessions,
  runner, file converter), `mimic-cli` (`convert`, `stream`, `models`).
- Export tooling (`tools/export`): static ONNX graphs from the original seed-vc tiny weights, fixtures for parity
  tests, a rewritten DiT step emitting ORT fused ops in fp16, an fp16 HiFT body with an overflow-safe Snake,
  manifest generation.
- Measured on one RTX A4000 (block 0.25 s, 10 flow steps, CFG 0.7): 48 ms per block single call
  (PyTorch streaming reference: 199 ms), 4 concurrent calls in real time; 6 steps: 5 calls.
- Parity against PyTorch: every stage within fp16 noise (relative L2 ≤ 4e-3), speaker similarity of outputs
  0.669 vs 0.647 for the PyTorch streaming reference.

## 2026-10-30 · 2.0 generator set

- `models/Mimicking-v2.0-lowlat`: our own generator (MimickingTrain `runs/Mimicking-v2.0`, 1.0 + K=4096 soft content codebook, fine-tuned 10k
  steps) exported with the low-latency geometry. The codebook is baked into `length_regulator*.onnx`, so the runtime is
  unchanged; manifest has a `content_codebook` field. Parity vs PyTorch: regulator 6.7e-4 (fp32) / 2.7e-3 (CUDA TF32), DiT 8e-4.
- Exporter (`mimictrain.export --train-config`) and parity tool (`MimickingTrain/tools/export_parity.py`) in MimickingTrain.
- MimickingLab prefers this set by default (`--models` overrides).
