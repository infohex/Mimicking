English | [中文](license-review.zh-CN.md)

# Licence review (2026-10-27)

> Update 2026-11-01: the 2.0 official release's toolchain no longer contains any GPL code (XLS-R / CAM++ / HiFT are exported by MimickingTrain directly from the Apache-2.0 weights); the `tools/export`, `ref/seed-vc` and `bench/` mentioned below were archived to `C:/Workspace/Rust/_archive/Mimicking-seedvc-2026-11-01` and are only fetched when a baseline set needs re-exporting or a baseline parity check is needed. The rest of the analysis below is the original text of 2026-10-27.
>
> Decision 2026-11-01: the library code is licensed under Apache-2.0 (`LICENSE` and `NOTICE` added, Cargo `license = "Apache-2.0"`). Item 2 below is therefore settled; the GPL-adjacent material it discusses is archived and not distributed.


Licences read item by item: the original seed-vc repository LICENSE, the four Hugging Face model cards, the CosyVoice and 3D-Speaker repository LICENSEs,
the ModelScope CAM++ model page, the RVC repository LICENSE, the NVIDIA CUDA EULA and cuDNN SLA, the licence fields of onnxruntime and all Rust dependencies.
This is an engineering-side summary, not legal advice; legal review is still required before commercial release, especially for item 1.

## Overview

| Component | How we use it | Licence | Provenance | Conclusion |
|---|---|---|---|---|
| seed-vc code (Plachtaa/seed-vc, archived 2025-11-21) | `ref/` read-only reference; the Python scripts in `tools/export` **import its modules** to export the graphs | **GPL-3.0** (repository LICENSE) | github.com/Plachtaa/seed-vc | The Rust library contains none of its code; but `tools/export` links GPL code, so if these scripts are published they must be under GPL-3.0 |
| DiT flow-matching estimator + length regulator weights `DiT_uvit_tat_xlsr_ema.pth` | exported as `dit_step*.onnx`, `length_regulator*.onnx` | **GPL-3.0** (the HF model card has only a license tag, no text, no training data description, no statement on commercial use) | huggingface.co/Plachta/Seed-VC | **Main risk**, see item 1 |
| XLS-R `wav2vec2-xls-r-300m` | exported as `xlsr*.onnx` | Apache-2.0 (HF model card; training data VoxPopuli, MLS, CommonVoice, BABEL, VoxLingua107, 436k hours) | huggingface.co/facebook/wav2vec2-xls-r-300m | Commercial use and redistribution allowed; attach the licence and notice |
| HiFT vocoder `hift.pt` | exported as `hift_f0.onnx`, `hift_body.onnx`, `hift_source.npz` | Apache-2.0 (HF CosyVoice-300M model card tag; the CosyVoice repository LICENSE is Apache-2.0; the file headers of `modules/hifigan/*.py` inside seed-vc are also marked Alibaba Apache-2.0) | huggingface.co/FunAudioLLM/CosyVoice-300M | Commercial use allowed. The model card's disclaimer "content for academic demonstration only" refers to the sample audio, not the weight licence |
| CAM++ `campplus_cn_common.bin` | exported as `campplus.onnx`; the test bench's quality probe | Apache-2.0 (HF funasr/campplus tag; the ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` licence field is Apache-2.0; the 3D-Speaker repository is Apache-2.0; the file headers of `modules/campplus/*.py` inside seed-vc are marked 3D-Speaker Apache-2.0) | ModelScope / FunASR | Commercial use allowed. Training data is "a large dataset of about 200 000 Chinese speakers", source not published |
| RVC (algorithmic source of real-time block splitting, crossfade, SOLA) | seed-vc's real-time logic is "modified from RVC"; our Rust is a rewrite from the algorithm | MIT (liujing04) | github.com/RVC-Project | No obligations; recommend noting it in the acknowledgements |
| onnxruntime-gpu 1.22 (3 DLLs) | distributed with the runtime directory | MIT | microsoft/onnxruntime | Attach the LICENSE |
| CUDA 12 runtime libraries (cudart, cublas, cublasLt, cufft) | distributed with the runtime directory (taken from torch/lib) | NVIDIA CUDA EULA | docs.nvidia.com/cuda/eula | Attachment A explicitly lists them as redistributable with an application; see the conditions in item 3 |
| cuDNN 9 (cudnn*.dll) | distributed with the runtime directory | NVIDIA cuDNN SLA | docs.nvidia.com/deeplearning/cudnn/sla | Runtime .dll files may be distributed with an application; same conditions as above |
| Rust dependencies (the library: ~110 crates) | statically linked into `mimic.exe` | all permissive: MIT / Apache-2.0 / BSD / ISC / Zlib / Unlicense; `unicode-ident` carries a Unicode-3.0 notice; `r-efi` offers three, take MIT | crates.io | Attach a NOTICE at release (generated with `cargo about` or `cargo license`) |
| Test bench dependencies (363 crates, not distributed) | local only | as above, plus BSL-1.0 (Boost), OFL-1.1 fonts, `self_cell` is Apache-2.0 OR GPL-2.0 (take Apache) | — | Nothing to do |
| Our own code | — | Apache-2.0 (decided 2026-11-01; `LICENSE`) | — | See item 2 |

## 1. The GPL-3.0 weights: the real problem

Facts:

- The seed-vc code repository is GPL-3.0; the weights page on Hugging Face has only a `gpl-3.0` tag, an empty README, no description of the training data,
  no statement on whether commercial use is allowed, no supplementary terms of any kind. The author distinguishes between "code licence" and "weight licence" neither on GitHub nor on HF.
- The `dit_step.onnx`, `dit_step_fused.onnx` and `length_regulator*.onnx` we export from these weights are another storage form of the same set of parameters
  (transposed, fp16, fused operators, numerically equivalent), not an independent creation.
- The repository is archived (2025-11-21) and the author no longer maintains it, but the licence does not lapse because of archiving.

How to understand it:

- The GPL was written for software, and whether "weights constitute a copyrightable work" is legally unsettled. But when distributing files tagged GPL-3.0, the safest
  assumption is: **distributing the weights (including our exported graphs) means complying with GPL-3.0** — commercial use is allowed, but recipients have the right to redistribute and modify, and the licence text
  and the "source" (for weights, the weights themselves) must accompany them.
- Our Rust program itself is not a derivative work of the weights: it is an independent implementation that reads the weights as data at runtime. GPL's copyleft reaches only "works based on the work",
  and a program that reads data generally does not count. Therefore **the program can stay MIT/Apache, the weights are a separate matter**. This is the common reading, but there is no case law and legal counsel may be more conservative.
- What would truly be found to be a GPL derivative work is the Python scripts under `tools/export`: they `import modules.flow_matching`,
  `import modules.diffusion_transformer` and other GPL modules, and rewrite the same forward pass as `DiTClean`. If these scripts are published, they must be under GPL-3.0.
  The file headers of the `modules/hifigan` and `modules/campplus` directories inside seed-vc are Apache-2.0 and are not part of this.

Possible routes:

1. **Distribute weights and program separately**: the program directory MIT/Apache, the model directory provided separately with the GPL-3.0 text and a provenance statement (the manifest already records it).
   Recipients hold GPL rights over the model directory. Suitable for internal deployment and for customers who do not mind the model being redistributable.
2. **Ask the author for a licence**: contact Plachta for a separate commercial licence for the weights or an explicit statement. The repository is archived; a reply is uncertain.
3. **Self-trained weights**: train with seed-vc's training code (GPL); **the output of the training program is not bound by the GPL** (GPL section 2: a program's output is covered only when
   it itself contains the program's content), and the resulting weights are ours. The cost is data and compute, and the licences of the training data themselves need attention.
4. **Replace the GPL parts**: the DiT and the length regulator are the only GPL source; XLS-R, HiFT and CAM++ are all Apache-2.0. Switching to an Apache/MIT-licensed
   flow-matching spectrogram generator makes the whole thing permissive, but it amounts to changing the model and the quality has to be re-evaluated.

Another practical risk has nothing to do with licensing: the training data of the tiny weights is not published, so it is impossible to tell whether it contains unauthorised voices. A commercial product needs
"target timbres come only from reference recordings authorised by the customer" written into its terms of use.

## 2. Our own code

- Status: Apache-2.0 since 2026-11-01 (`LICENSE`, `NOTICE`, Cargo metadata). The two options below were the original analysis.
- If MIT/Apache is chosen: the repository must mark `tools/export` separately as GPL-3.0 (or not ship it with the library, internal use only), stated in the README.
- If the whole repository is GPL-3.0: the simplest and most consistent (same licence as tools/export and the weights), at the cost that every integrator must open-source their whole.
- Either way, the release package must include: our LICENSE, the MIT text of onnxruntime, the Apache-2.0 texts and NOTICEs of XLS-R/HiFT/CAM++,
  the aggregated NOTICE of the Rust dependencies, the CUDA/cuDNN EULA pointers, the RVC acknowledgement.

## 3. Distribution conditions of the NVIDIA runtime libraries

Attachment A of the CUDA EULA explicitly lists the cudart, cublas, cublasLt and cufft DLLs as redistributable with an application; the cuDNN SLA allows distributing the runtime .dll files. Shared conditions:

- Distribution only as **part of our application**, not standalone and not as an SDK; the application must have substantial functionality (satisfied).
- Only our application accesses these libraries (the runtime directory layout satisfies this: `runtime/cuda` is added to PATH only by `mimic.exe`).
- Our distribution terms must not conflict with the EULA and must protect NVIDIA's intellectual property; reverse engineering and removal of copyright notices are prohibited;
  **"must not be used in a way that would subject the SDK to an open-source licence"** — this clause interacts with item 1: if the whole release package is distributed under GPL-3.0, putting NVIDIA's
  DLLs in the same GPL package is contentious. The safe approach: the runtime directory (program + CUDA/cuDNN/ORT) distributed under our permissive licence or proprietary terms,
  the model directory distributed separately under the GPL; or do not bundle CUDA and require the target machine to install the CUDA 12 / cuDNN 9 runtime libraries.
- Not for "critical applications" (aviation, medical, autonomous driving and the like).
- Taking the DLLs from `torch/lib` is only the means of acquisition; the licence of these files remains NVIDIA's, not torch's BSD-3.

## 4. The remaining components: no obstacles

- The three Apache-2.0 weights XLS-R, HiFT and CAM++: commercial use, modification and redistribution allowed; the obligations are to attach the licence text, keep the NOTICE and state the modifications.
  Our exported ONNX graphs count as "modifications"; stating this in the manifest and the NOTICE is enough.
- onnxruntime MIT, Rust dependencies all permissive.
- RVC MIT: seed-vc's real-time window logic comes from it and we rewrote it from the algorithm; an acknowledgement is enough.

## 5. Suggested actions

1. BO decides which route item 1 takes (separate distribution / ask for a licence / self-train / change model); this is a product decision.
2. Before that: the repository README's licence section states the three categories (our code, tools/export as GPL-3.0, models under their respective original licences);
   `make_manifest.py` already writes the licences of all four weights into the manifest (CAM++ and the RVC acknowledgement added this time).
3. The packaging script `package_runtime.py` should in future generate `THIRD-PARTY-NOTICES`: ORT MIT, the three Apache-2.0, the Rust dependency aggregate, the NVIDIA EULA pointers.
4. The sentence "model licence is GPL-3.0, needs legal confirmation or switching to self-trained weights" in the external documentation (the tour page) stays as it is.

## 6. Feasibility of self-trained weights (assessed 2026-10-27)

Facts after reading `train.py`:

- Training only updates `cfm` (the DiT) and `length_regulator`; XLS-R, CAM++ and HiFT are frozen and all Apache-2.0. Exactly the two GPL parts can be self-trained.
- Training from scratch is supported: `pretrained_model: ""` in the config and no `--pretrained-ckpt` means random initialisation. Fine-tuning from the GPL checkpoint is **not** an option; that would still be a derivative.
- Training uses OpenVoice (MIT, including checkpoints_v2) for timbre-perturbation augmentation and downloads `se_db.pt` (a speaker embedding bank, GPL tag) from Plachta/Seed-VC
  as the perturbation targets. `se_db` is only a training input and does not enter the weights; to be safe, rebuild one from our own data with OpenVoice's `extract_se`.
- The output (weights) of a GPL training program is not bound by the GPL (GPL-3.0 section 2); the architecture itself is not copyrightable; the Rust inference side and the export scripts need no changes.
- Data requirements: 1–30 s of clean multi-speaker speech, no labels needed; the data's licence decides whether the result can be used commercially:
  LibriTTS-R (CC BY 4.0, 585 h English), MLS (CC BY 4.0, 8 languages, tens of thousands of hours of English), AISHELL-3 (Apache-2.0, 85 h Chinese, 218 speakers) are usable commercially;
  Emilia (CC BY-NC 4.0) and WenetSpeech4TTS (terms of use restrict to non-commercial) are not.
- Compute: the v1 `train.py` is single-GPU; 25M parameters, batch 2 × 100 s, running the frozen XLS-R twice per step (raw + perturbed) is the bulk; estimated 0.5–1 s/step on an A4000,
  training from scratch at 300 000–1 000 000 steps is 3–10 days on one card, 16 GB is enough; two cards can run two experiments in parallel.
- The risk is in quality, not engineering: the training data of the public tiny weights is unknown, and the similarity and intelligibility of the self-trained version must be re-evaluated on the test bench; commercial Chinese data is the bottleneck
  (AISHELL-3 is too small; licensed data would have to be bought or collected).
