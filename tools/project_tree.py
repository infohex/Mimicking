"""Generate docs/project-tree.html (English) and docs/project-tree.zh-CN.html (Chinese): an interactive, annotated
map of the whole Mimicking project (this repository plus the sibling workspaces MimickingTrain and MimickingLab,
and the data root).

    python tools/project_tree.py                 # writes both pages (no dependencies beyond the stdlib)
    python tools/project_tree.py --lang en       # English page only; --lang zh-CN for the Chinese page; --lang all (default)

What it records per entry: path, kind, size (recursive for directories), file count, the first documentation
line of source files (`//!` in Rust, the module docstring in Python, the first heading in Markdown, skipping a
leading `English | 中文` language-switch line), the git state inside this repository (tracked / modified /
untracked / ignored), and the hand-written description in `DESC` below. Every `DESC` entry is bilingual:
`title` / `desc` are English, `title_zh` / `desc_zh` Chinese; tags use the Chinese words as internal keys and are
rendered through `TAG_LABELS`. Files named `X.zh-CN.<ext>` with no `DESC` entry get an automatic description
("Chinese version of X.<ext>") pointing at the English sibling. All UI strings live in `I18N`; the two pages are
the same template rendered per language, each linking to the other next to the title. Directories with many
generated files (model sets, runs, references, outputs) are listed one level deep and summarised. Edit `DESC`
when the layout changes and re-run; the pages are self-contained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # Mimicking/
WS = HERE.parent                                       # C:/Workspace/Rust
ROOTS = [("Mimicking", HERE), ("MimickingTrain", WS / "MimickingTrain"), ("MimickingLab", WS / "MimickingLab")]
DATA_ROOT = Path("C:/MimickingData")
OUT = {"en": HERE / "docs" / "project-tree.html", "zh-CN": HERE / "docs" / "project-tree.zh-CN.html"}
LANGS = ("en", "zh-CN")

EXCLUDE = {"target", ".venv", ".git", "__pycache__", ".pytest_cache", "node_modules", ".cargo", ".pytest_cache"}
# directories listed one level deep: children are summarised (file count + size), not expanded
SHALLOW = {
    "Mimicking/ref", "Mimicking/models", "Mimicking/out", "Mimicking/dist",
    "MimickingTrain/runs", "MimickingTrain/docs/research", "MimickingLab/out", "MimickingLab/crates/mimic-lab/refs_byname",
}

# tags: the Chinese words are the internal keys (also used as CSS / colour keys); the page shows TAG_LABELS[lang][key]
TAG_LABELS = {
    "源码": "source", "工具": "tools", "文档": "docs", "模型": "models", "数据": "data", "记录": "records",
    "第三方": "third-party", "构建": "build", "配置": "config", "测试": "tests", "正式版": "release",
}
CATS = ["源码", "工具", "文档", "模型", "数据", "记录", "第三方", "构建", "配置", "测试"]  # filter chips (正式版 is a tag only)

# ------------------------------------------------------------------ hand-written descriptions
# path -> title / desc (English), title_zh / desc_zh (Chinese), tags (keys of TAG_LABELS), related (paths relative to the root)
DESC: dict[str, dict] = {
    # ---------------- Mimicking (library)
    "Mimicking": dict(
        title="Library repository (git)", title_zh="库仓库（git）", tags=["源码"],
        desc=("The Rust real-time voice conversion library itself: six crates, the command line, examples, export tools, docs, the models "
              "directory. Dependencies point top-down: mimic-core / mimic-pipeline depend on no model or runtime; mimic-model is the first "
              "model package. The 2.0 official release notes are in docs/release-2.0.md, integration in docs/integration-guide.md."),
        desc_zh=("Rust 实时变声库本体：六个 crate、命令行、示例、导出工具、文档、模型目录。依赖方向自上而下：mimic-core / mimic-pipeline 不依赖任何模型或运行时，"
                 "mimic-model 是第一个模型包。2.0 正式版的说明见 docs/release-2.0.md，接入见 docs/integration-guide.md。"),
        related=["docs/release-2.0.md", "docs/integration-guide.md", "README.md", "CLAUDE.md"]),
    "Mimicking/Cargo.toml": dict(
        title="Workspace manifest", title_zh="workspace 清单", tags=["配置"],
        desc="Workspace of six crates + examples; the ort crate talks to onnxruntime 1.22 via load-dynamic + api-22, CUDA feature enabled.",
        desc_zh="六个 crate + 示例的 workspace；ort crate 以 load-dynamic + api-22 对接 onnxruntime 1.22，CUDA 特性开启。"),
    "Mimicking/CLAUDE.md": dict(
        title="Project rules", title_zh="项目规则", tags=["文档"],
        desc=("Hard rules for contributors (AI included): one ORT environment per process, no concurrency during CUDA Graph capture, one thread "
              "per GPU, static shapes read from the manifest, codebook models exported with --norm-stats, Silero gating by default, queue steered "
              "by playback rate, etc."),
        desc_zh="给协作者（含 AI）的硬规则：一个进程一个 ORT 环境、CUDA Graph 录制期不并发、一卡一线程、静态形状从 manifest 读、码本模型导出带 --norm-stats、门控默认 Silero、队列用速率微调等。"),
    "Mimicking/LICENSE": dict(
        title="Apache-2.0 licence", title_zh="Apache-2.0 许可证", tags=["文档"],
        desc="Full text of the Apache License 2.0; the library code is licensed under it since 2026-11-01 (NOTICE carries the copyright line).",
        desc_zh="Apache License 2.0 全文；库代码自 2026-11-01 起采用此许可（版权行在 NOTICE）。"),
    "Mimicking/NOTICE": dict(
        title="Copyright notice", title_zh="版权声明", tags=["文档"],
        desc="Apache-2.0 NOTICE file: project name and copyright holder.", desc_zh="Apache-2.0 的 NOTICE 文件：项目名与版权人。"),
    "Mimicking/CHANGELOG.md": dict(
        title="Changelog", title_zh="变更记录", tags=["文档"],
        desc="Recorded per version; 2.0 — 2026-11-01 is the official release entry.",
        desc_zh="按版本记录；2.0 — 2026-11-01 为正式版条目。"),
    "Mimicking/README.md": dict(
        title="Overview and quick start", title_zh="总览与快速开始", tags=["文档"],
        desc="Structure, quick-start code, model and runtime-library directory conventions, how to test, performance numbers, licence.",
        desc_zh="结构、快速开始代码、模型与运行库目录约定、测试方法、性能数字、许可。"),
    "Mimicking/.github": dict(
        title="CI", title_zh="CI", tags=["配置"],
        desc="GitHub Actions: cargo build / test (the parts that need no GPU).",
        desc_zh="GitHub Actions：cargo build / test（无 GPU 的部分）。"),
    "Mimicking/crates": dict(
        title="Six crates", title_zh="六个 crate", tags=["源码"],
        desc="core (traits) → dsp (pure-Rust signal processing) / onnx (ORT backend) → pipeline (session and solver) → model (model package) → cli.",
        desc_zh="core（trait）→ dsp（纯 Rust 信号处理）/ onnx（ORT 后端）→ pipeline（会话与求解器）→ model（模型包）→ cli。"),
    "Mimicking/crates/mimic-core": dict(
        title="Stage traits", title_zh="阶段 trait", tags=["源码"],
        desc=("The Graph, ContentEncoder, SpeakerEncoder, LengthRegulator, VelocityEstimator, FlowStepper, Vocoder, MelAnalyzer, Resampler, Vad "
              "interfaces and the Prompt / Error types. Depends on no model or runtime."),
        desc_zh="Graph、ContentEncoder、SpeakerEncoder、LengthRegulator、VelocityEstimator、FlowStepper、Vocoder、MelAnalyzer、Resampler、Vad 等接口与 Prompt / Error 类型。不依赖模型或运行时。"),
    "Mimicking/crates/mimic-dsp": dict(
        title="Pure-Rust DSP", title_zh="纯 Rust DSP", tags=["源码"],
        desc=("Resampling (same algorithm as torchaudio sinc_interp_hann), mel spectrogram, Kaldi fbank, the host-side part of HiFT (harmonic "
              "source, STFT/ISTFT), SOLA stitching, energy VAD, Gaussian noise. All have parity tests."),
        desc_zh="重采样（torchaudio sinc_interp_hann 同算法）、梅尔谱、Kaldi fbank、HiFT 的主机侧部分（谐波源、STFT/ISTFT）、SOLA 拼接、能量 VAD、高斯噪声。全部有对拍测试。"),
    "Mimicking/crates/mimic-dsp/src/vad.rs": dict(
        title="Energy gate (fallback)", title_zh="能量门（退路）", tags=["源码"],
        desc=("Loudness gate with adaptive floor + hysteresis. The floor is initialised from the first block and gets pinned on trimmed files and "
              "on leading digital silence; it cannot separate continuous typing from speech. The official release switches the default to "
              "mimic-onnx's SileroVad; this gate is only the fallback when the model is missing."),
        desc_zh="自适应地板 + 滞回的响度门。地板以第一块初始化，对剪好的文件和数字静音开头会钉死；分不开连续打字与说话。正式版默认改用 mimic-onnx 的 SileroVad，此门仅作缺模型时的退路。"),
    "Mimicking/crates/mimic-dsp/src/sola.rs": dict(
        title="SOLA stitching", title_zh="SOLA 拼接", tags=["源码"],
        desc="Searches ±20 ms between blocks for the best alignment, then a 40 ms crossfade; 60 ms of the 80 ms algorithmic delay comes from here.",
        desc_zh="块与块之间 ±20 ms 搜索最佳对齐后 40 ms 交叉淡化；算法固有延迟 80 ms 里的 60 ms 来自这里。"),
    "Mimicking/crates/mimic-dsp/tests": dict(
        title="DSP parity", title_zh="DSP 对拍", tags=["测试"],
        desc="Stage-by-stage parity against the PyTorch fixtures in <set>/fixtures (MIMIC_MODELS / MIMIC_FIXTURES; skipped when absent); bench_hift measures the host-side HiFT cost.",
        desc_zh="对套内 fixtures/ 的 PyTorch 夹具逐级对拍（MIMIC_MODELS / MIMIC_FIXTURES；无夹具时跳过）；bench_hift 量主机侧 HiFT 成本。"),
    "Mimicking/crates/mimic-onnx": dict(
        title="ONNX Runtime backend", title_zh="ONNX Runtime 后端", tags=["源码"],
        desc=("Loads onnxruntime.dll at runtime (no link-time dependency), OnnxGraph (plain session), BoundGraph (device-resident inputs/outputs + "
              "CUDA Graph replay), the process-wide environment and capture lock, Silero VAD."),
        desc_zh="运行时加载 onnxruntime.dll（无链接期依赖）、OnnxGraph（普通会话）、BoundGraph（设备驻留输入输出 + CUDA Graph 回放）、进程唯一的环境与录制锁、Silero VAD。"),
    "Mimicking/crates/mimic-onnx/src/vad.rs": dict(
        title="Silero VAD gate (default)", title_zh="Silero VAD 门（默认）", tags=["源码", "正式版"],
        desc=("Silero v6 ONNX (MIT) wired to the Vad trait: 64-sample context + 512-sample chunks, state (2,1,128); any slice in the block > 0.5 "
              "opens the gate, 2 consecutive blocks < 0.35 close it, < −60 dBFS is silenced outright. Keyboard typing / desk tapping: 0 gate "
              "openings; clean speech: full recall. The model file is in models/vad."),
        desc_zh="Silero v6 ONNX（MIT）接到 Vad trait：64 点上下文 + 512 点块、状态 (2,1,128)；块内任一片 > 0.5 开门，连续 2 块 < 0.35 关门，< −60 dBFS 直接静音。键盘 / 敲桌 0 开门，干净语音全召回。模型文件在 models/vad。"),
    "Mimicking/crates/mimic-pipeline": dict(
        title="Session and solver", title_zh="会话与求解器", tags=["源码"],
        desc=("Stages (the trait objects of each stage), StreamGeometry (window geometry), Session (per-stream rolling window, SOLA, noise seed, "
              "prompt, gating and the latency-aligned gate queue), Runner::push (one block in, one block out), the EulerCfg solver, Converter "
              "(whole files through the streaming path)."),
        desc_zh="Stages（各阶段 trait 对象）、StreamGeometry（窗口几何）、Session（每路的滚动窗口、SOLA、噪声种子、提示、门控与按延迟对齐的门控队列）、Runner::push（一块进一块出）、EulerCfg 求解器、Converter（整文件走流式路径）。"),
    "Mimicking/crates/mimic-model": dict(
        title="Model package (the XLS-R → length regulator → DiT → HiFT family)", title_zh="模型包（XLS-R → 调节器 → DiT → HiFT 这一族）", tags=["源码"],
        desc=("Loads a set of static graphs per manifest and implements the traits: XLS-R content encoding, CAM++ style, length regulation, fused "
              "DiT stepping, HiFT; WindowConfig::shapes window arithmetic; ModelFiles / Manifest "
              "validation."),
        desc_zh="按 manifest 加载一套静态图并实现各 trait：XLS-R 内容编码、CAM++ 风格、长度调节、DiT 融合步进、HiFT；WindowConfig::shapes 窗口算术；ModelFiles / Manifest 校验。"),
    "Mimicking/crates/mimic-model/tests": dict(
        title="Model package parity and concurrent-load tests", title_zh="模型包对拍与并发加载测试", tags=["测试"],
        desc="Need a GPU and MIMIC_ORT_DYLIB: stage-by-stage parity against the PyTorch fixtures in the set, the 10-step solver, simultaneous multi-thread multi-GPU loading (CUDA Graph capture lock).",
        desc_zh="需要 GPU 与 MIMIC_ORT_DYLIB：逐级对拍套内夹具、10 步求解器、多线程多卡同时加载（CUDA Graph 录制锁）。"),
    "Mimicking/crates/mimic-cli": dict(
        title="The mimic command line", title_zh="命令行 mimic", tags=["源码"],
        desc=("convert (file to file), stream (N-stream real-time simulation, with --realtime and the --vad energy gate), models list / verify. "
              "Without --models it takes the first set with a manifest under <exe>/models or ./models (*-lowlat preferred)."),
        desc_zh="convert（文件到文件）、stream（N 路实时模拟，可 --realtime、--vad 能量门）、models list / verify。不带 --models 时取 <exe>/models 或 ./models 下第一个带 manifest 的套（优先 *-lowlat）。"),
    "Mimicking/examples/convert-minimal": dict(
        title="Minimal example", title_zh="最小示例", tags=["源码"],
        desc="A complete conversion program in under 50 lines: load, prompt, session, Converter.",
        desc_zh="50 行以内的完整转换程序：加载、提示、会话、Converter。"),
    "Mimicking/tools": dict(
        title="Python tools", title_zh="Python 工具", tags=["工具"],
        desc=("package_runtime.py assembles the runtime-library directory from .venv; project_tree.py generates this page."),
        desc_zh="package_runtime.py 从 .venv 装配运行库目录；project_tree.py 生成本页。"),
    "Mimicking/tools/project_tree.py": dict(
        title="Generator of this page", title_zh="本页生成器", tags=["工具"],
        desc=("Scans the three workspaces and the data root, merges the hand-written descriptions, writes docs/project-tree.html (English) and "
              "docs/project-tree.zh-CN.html (Chinese). When the layout changes, edit DESC and run it again."),
        desc_zh="扫描三个工作区与数据根，合并手写说明，写出 docs/project-tree.html（英文）与 docs/project-tree.zh-CN.html（中文）。目录变了改 DESC 再跑一次。"),
    "Mimicking/models": dict(
        title="Models directory (not in git)", title_zh="模型目录（不进 git）", tags=["模型"],
        desc=("One set of static graphs + manifest.json (geometry, sha256, provenance) per subdirectory: Mimicking-v2.0* (self-trained "
              "weights, codebook and frozen length-regulator statistics baked in, graph version 2.0.1); the other subdirectories are local "
              "evaluation references only (not distributed)."),
        desc_zh=("每个子目录一套静态图 + manifest.json（几何、sha256、来源）：Mimicking-v2.0*（自训权重，码本与冻结的调节器统计量已烘入，图版本 2.0.1）；"
                 "其他子目录只是本地评测用的对照套（不分发）。"),
        related=["docs/release-2.0.md"]),
    "Mimicking/models/Mimicking-v2.0-lowlat": dict(
        title="2.0 official release default set (240 ms)", title_zh="2.0 正式版默认套（240 ms）", tags=["模型", "正式版"],
        desc="Block 0.25 s (240 ms in practice), right context 0.02 s, prompt 3 s; loaded by the Lab by default. Streaming WER on the 32 pairs: 7.6 %.",
        desc_zh="块 0.25 s（实际 240 ms）、右前瞻 0.02 s、提示 3 s；Lab 默认加载。流式 32 对 WER 7.6 %。"),
    "Mimicking/models/Mimicking-v2.0-b0.16-r0.02": dict(
        title="2.0 set, 160 ms block", title_zh="2.0 套，160 ms 块", tags=["模型", "正式版"],
        desc="Latency first: end-to-end about 330 ms, GPU about 19 %.",
        desc_zh="延时优先：端到端约 330 ms，GPU 约 19 %。"),
    "Mimicking/models/Mimicking-v2.0-b0.32-r0.02": dict(
        title="2.0 set, 320 ms block", title_zh="2.0 套，320 ms 块", tags=["模型", "正式版"],
        desc="Compute first: GPU about 9 %, 80 ms more latency.",
        desc_zh="算力优先：GPU 约 9 %，延时多 80 ms。"),
    "Mimicking/models/onnx-lowlat": dict(
        title="Reference set, 240 ms / 0.02 s", title_zh="对照套，240 ms / 0.02 s", tags=["模型", "第三方"],
        desc="Local evaluation reference (earlier model weights, not distributed); also the exporter's --shared base fallback source for this geometry.",
        desc_zh="本地评测用的对照套（早期模型权重，不分发）；也是导出器 --shared base 退路在该几何下的来源。"),
    "Mimicking/models/onnx-b0.16-r0.02": dict(
        title="Reference set, 160 ms", title_zh="对照套，160 ms", tags=["模型", "第三方"],
        desc="Local evaluation reference for 160 ms blocks (the window length of xlsr.onnx changes with the block length); --shared base fallback for this geometry.",
        desc_zh="160 ms 块的本地对照套（xlsr.onnx 的窗口长度随块长变）；该几何下 --shared base 的退路。"),
    "Mimicking/models/onnx-b0.32-r0.02": dict(
        title="Reference set, 320 ms", title_zh="对照套，320 ms", tags=["模型", "第三方"],
        desc="Local evaluation reference for 320 ms blocks; --shared base fallback for this geometry.",
        desc_zh="320 ms 块的本地对照套；该几何下 --shared base 的退路。"),
    "Mimicking/models/onnx-b0.25-r0.10": dict(
        title="Reference set, right context 0.10 s", title_zh="对照套，右前瞻 0.10 s", tags=["模型", "第三方"],
        desc="Reference for the right-context experiment (conclusion: no benefit for 2.0); kept as the geometry source for re-exports.",
        desc_zh="右前瞻实验的对照（结论：对 2.0 无益）；留作重导的几何源。"),
    "Mimicking/models/vad": dict(
        title="Silero VAD model", title_zh="Silero VAD 模型", tags=["模型", "第三方", "正式版"],
        desc="silero_vad.onnx (v6 16 kHz op15, 1.2 MB, MIT) + licence text. When it is missing, that Lab stream falls back to the energy gate and reports an error.",
        desc_zh="silero_vad.onnx（v6 16 kHz op15，1.2 MB，MIT）+ 许可文本。缺失时 Lab 的该路退回能量门并报错。"),
    "Mimicking/ref": dict(
        title="Reference weights (not in git)", title_zh="参考权重（不进 git）", tags=["第三方", "数据"],
        desc=("models/: the Apache-2.0 weights used by training evaluation and export; sv_en/: English VoxCeleb CAM++ (the second yardstick of the "
              "evaluation)."),
        desc_zh="models/：训练评估与导出用的 Apache-2.0 权重；sv_en/：英文 VoxCeleb CAM++（评测的第二把尺子）。"),
    "Mimicking/ref/models": dict(
        title="Apache-2.0 pretrained weights", title_zh="Apache-2.0 预训练权重", tags=["第三方", "数据", "正式版"],
        desc=("hift.pt (CosyVoice HiFT) and campplus_cn_common.bin (CAM++), each with licence and NOTICE; the README records provenance and sha256. "
              "Training evaluation and the mimictrain.export.shared export both read from here; XLS-R is loaded from the user cache by its HF name."),
        desc_zh="hift.pt（CosyVoice HiFT）与 campplus_cn_common.bin（CAM++），各带许可与 NOTICE，README 记来源与 sha256。训练评估与 mimictrain.export.shared 的导出都从这里读；XLS-R 按 HF 名字从用户缓存加载。"),
    "Mimicking/ref/sv_en": dict(
        title="English speaker model", title_zh="英文说话人模型", tags=["第三方", "数据"],
        desc="campplus_voxceleb.bin: an English CAM++ unrelated to training, the second yardstick for the \"similarity to target / similarity to source\" evaluation.",
        desc_zh="campplus_voxceleb.bin：与训练无关的英文 CAM++，评测\"像目标 / 像源\"的第二把尺子。"),
    "Mimicking/docs": dict(
        title="Docs", title_zh="文档", tags=["文档"],
        desc=("release-2.0 (official release), integration-guide (integration), runtime-pitfalls (runtime rules), capacity-a4000 (capacity), "
              "license-review, history/ (replication-phase records and plans), two interactive diagrams (overview, pipeline-flow) and this page. "
              "Each document has a .zh-CN Chinese sibling."),
        desc_zh="release-2.0（正式版）、integration-guide（接入）、runtime-pitfalls（运行时规则）、capacity-a4000（容量）、license-review、history/（复刻阶段的记录与计划）、两张交互图（overview、pipeline-flow）与本页。每篇都有 .zh-CN 中文版。"),
    "Mimicking/docs/history": dict(
        title="History (replication phase)", title_zh="历史记录（复刻阶段）", tags=["文档", "记录"],
        desc="M0–M3 status reports, export-side GPU optimisation, the original plan and the training plan (October 2026, 0.1.0-era numbers); engineering record only.",
        desc_zh="M0–M3 阶段报告、导出侧 GPU 优化、最初的总体计划与训练计划（2026 年 10 月，数字为 0.1.0 时代）；只作工程记录。"),
    "Mimicking/docs/release-2.0.md": dict(
        title="2.0 official release notes", title_zh="2.0 正式版说明", tags=["文档", "正式版"],
        desc="Frozen manifest and checksums, final parameters (model / training / export / runtime), offline and streaming numbers, weights and interface, known limitations, reproduction commands.",
        desc_zh="冻结清单与校验和、最终参数（模型 / 训练 / 导出 / 运行时）、离线与流式数字、权重与接口、已知限制、复现命令。"),
    "Mimicking/docs/integration-guide.md": dict(
        title="Integration guide", title_zh="接入指南", tags=["文档", "正式版"],
        desc="For projects integrating the library: runtime environment, choosing a set, minimal code, audio pipeline and queue control, latency budget, gating, capacity, acceptance, FAQ.",
        desc_zh="接入本库的项目看这篇：运行环境、选套、最小代码、音频管线与队列控制、延时预算、门控、容量、验收、常见问题。"),
    "Mimicking/docs/runtime-pitfalls.md": dict(
        title="Runtime pitfalls and rules", title_zh="运行时的坑与规则", tags=["文档"],
        desc="ORT environment, CUDA Graph, threads, static shapes, GPU downclocking, gate alignment, codebook statistics, energy gate, queue playback-rate steering, microphone AGC.",
        desc_zh="ORT 环境、CUDA Graph、线程、静态形状、GPU 降频、门控对齐、码本统计量、能量门、队列速率微调、麦克风 AGC。"),
    "Mimicking/docs/pipeline-flow.html": dict(
        title="Pipeline flow (interactive, 2.0 official release)", title_zh="流程图（交互，2.0 正式版）", tags=["文档", "正式版"],
        desc=("Top-down tree flow: the real-time chain (with gating and vocoder branches), the reference prompt, where the models come from, "
              "real-time system threads and queues, milestones and the latency / quality table, glossary. Click a card for inputs / outputs / role / destination."),
        desc_zh="自上而下的树状流程：实时链路（含门控与声码器分支）、参考提示、模型从哪里来、实时系统线程与队列、里程碑与延时 / 质量表、术语表。点卡片看输入 / 输出 / 作用 / 去向。"),
    "Mimicking/docs/overview.html": dict(
        title="Project overview (interactive)", title_zh="项目导览（交互）", tags=["文档"],
        desc="The project-level overview page.",
        desc_zh="项目层面的导览页。"),
    "Mimicking/docs/project-tree.html": dict(
        title="This page: project tree", title_zh="本页：目录结构", tags=["文档"],
        desc="Generated by tools/project_tree.py.",
        desc_zh="由 tools/project_tree.py 生成。"),
    "Mimicking/dist": dict(
        title="Packaged output (not in git)", title_zh="打包产物（不进 git）", tags=["构建"],
        desc="The self-contained runtime directory assembled by package_runtime.py (mimic.exe + runtime/ + models/).",
        desc_zh="package_runtime.py 装配出的自包含运行目录（mimic.exe + runtime/ + models/）。"),
    "Mimicking/out": dict(
        title="CLI output (not in git)", title_zh="CLI 输出（不进 git）", tags=["构建"],
        desc="Output wav files of mimic convert / stream.",
        desc_zh="mimic convert / stream 的输出 wav。"),
    # ---------------- MimickingTrain
    "MimickingTrain": dict(
        title="Training and export (pure Python, not in git)", title_zh="训练与导出（纯 Python，不在 git）", tags=["源码"],
        desc=("Our own definition of the generator (DiT + length regulator), data pipeline, trainer, evaluation, ONNX export, diagnostic tools. "
              "No GPL code. The official release weights are in runs/Mimicking-v2.0."),
        desc_zh="生成器（DiT + 长度调节器）的自有定义、数据流水线、训练器、评估、ONNX 导出、诊断工具。不含 GPL 代码。正式版权重在 runs/Mimicking-v2.0。",
        related=["README.md", "docs/lessons-learned.md", "docs/despeaker-design.md"]),
    "MimickingTrain/README.md": dict(
        title="Status, data, training, export usage", title_zh="状态、数据、训练、导出用法", tags=["文档"],
        desc=("2.0 official release status, the T1 data pipeline, Common Voice augmentation, the trainer, two-GPU training, evaluation and export "
              "commands (including the 2.0 recipe with --train-config / --norm-stats)."),
        desc_zh="2.0 正式版状态、T1 数据流水线、Common Voice 扩充、训练器、两卡训练、评估与导出命令（含 --train-config / --norm-stats 的 2.0 配方）。"),
    "MimickingTrain/configs": dict(
        title="Training configs", title_zh="训练配置", tags=["配置"],
        desc=("train_tiny_sp (1.0 from scratch, 100 k steps), ft_control / ft_adv / ft_cv / ft_cv_adv / ft_selfconv* (despeaker comparison fine-tunes), "
              "ft_vq4096 (the 2.0 official fine-tune), ft_vq4096_30k (the stopped continuation)."),
        desc_zh="train_tiny_sp（1.0 从零 100 k 步）、ft_control / ft_adv / ft_cv / ft_cv_adv / ft_selfconv*（去说话人化对照微调）、ft_vq4096（2.0 正式微调）、ft_vq4096_30k（已停的续训）。"),
    "MimickingTrain/configs/ft_vq4096.json": dict(
        title="Fine-tune config of 2.0", title_zh="2.0 的微调配置", tags=["配置", "正式版"],
        desc="Warm-started from the 1.0 EMA, K=4096 soft codebook, 10 k steps, lr 5e-5 cosine, 24 000 frames / step on each of two GPUs.",
        desc_zh="从 1.0 EMA 热启动，K=4096 软码本，10 k 步，lr 5e-5 余弦，两卡各 24 000 帧 / 步。"),
    "MimickingTrain/mimictrain": dict(
        title="Training package", title_zh="训练包", tags=["源码"],
        desc=("model (DiT + length regulator, with freeze_norm_stats), cfm (flow-matching loss and sampler), codebook (content codebook), geometry "
              "(window arithmetic matching Rust), upstream (reference checkpoint mapping, parity only), data/, train/, export/, thirdparty/."),
        desc_zh="model（DiT + 调节器，含 freeze_norm_stats）、cfm（流匹配损失与采样器）、codebook（内容码本）、geometry（与 Rust 一致的窗口算术）、upstream（参考检查点映射，仅对拍）、data/、train/、export/、thirdparty/。"),
    "MimickingTrain/mimictrain/model.py": dict(
        title="Generator definition", title_zh="生成器定义", tags=["源码", "正式版"],
        desc="DiT: 384-dim, 9 layers, 6 heads, RoPE, SwiGLU, U-ViT; LengthRegulator (Conv3 + GroupNorm + Mish ×4) and freeze_norm_stats for inference.",
        desc_zh="DiT 384 维 9 层 6 头 RoPE SwiGLU U-ViT；LengthRegulator（Conv3 + GroupNorm + Mish ×4）及推理用的 freeze_norm_stats。"),
    "MimickingTrain/mimictrain/codebook.py": dict(
        title="Content codebook", title_zh="内容码本", tags=["源码", "正式版"],
        desc="Soft / hard backfill through a frozen k-means codebook, between XLS-R and the length regulator; baked into the regulator graph at export.",
        desc_zh="冻结 k-means 码本的软 / 硬回填，XLS-R 与调节器之间；导出时烘进调节器图。"),
    "MimickingTrain/mimictrain/geometry.py": dict(
        title="Window geometry", title_zh="窗口几何", tags=["源码"],
        desc="The block / window / frame-count arithmetic that must match mimic-model WindowConfig::shapes.",
        desc_zh="与 mimic-model WindowConfig::shapes 必须一致的块 / 窗口 / 帧数算术。"),
    "MimickingTrain/mimictrain/data": dict(
        title="Data pipeline", title_zh="数据流水线", tags=["源码"],
        desc=("LibriTTS-R download and manifests, Common Voice manifests, feature cache (int16 waveform + fp16 mel + CAM++ vector), timbre perturbation, "
              "self-conversion perturbation, CAM++ (PyTorch)."),
        desc_zh="LibriTTS-R 下载与清单、Common Voice 清单、特征缓存（int16 波形 + fp16 梅尔 + CAM++ 向量）、音色扰动、自转换扰动、CAM++（PyTorch）。"),
    "MimickingTrain/mimictrain/train": dict(
        title="Trainer and evaluation", title_zh="训练器与评估", tags=["源码"],
        desc=("Config dataclass, length-bucketed dataset, single step (flow-matching loss), EMA, bf16 + DDP(gloo) training loop, validation_loss and "
              "Synth (32-pair synthesis evaluation), HiFT for evaluation."),
        desc_zh="配置 dataclass、长度分桶数据集、单步（流匹配损失）、EMA、bf16 + DDP(gloo) 训练循环、validation_loss 与 Synth（32 对合成评估）、评估用 HiFT。"),
    "MimickingTrain/mimictrain/export": dict(
        title="ONNX exporter (fully self-contained)", title_zh="ONNX 导出器（整套自给自足）", tags=["源码", "正式版"],
        desc=("dit_step / dit_step_fused (fp16 fused) and length_regulator* (fp32, codebook and frozen statistics baked in) are exported by __main__; "
              "shared.py exports xlsr* / campplus / hift_* from the Apache weights in ref/models and writes the Rust parity fixtures to fixtures/ "
              "inside the set; after export all graph I/O is checked to be fp32; the manifest carries provenance. --shared base can still hard-link."),
        desc_zh="dit_step / dit_step_fused（fp16 融合）、length_regulator*（fp32，烘码本与冻结统计量）由 __main__ 导；shared.py 从 ref/models 的 Apache 权重导 xlsr* / campplus / hift_* 并写 Rust 对拍夹具到套内 fixtures/；导出后检查所有图 I/O 为 fp32；manifest 带 provenance。--shared base 仍可硬链接。"),
    "MimickingTrain/mimictrain/thirdparty": dict(
        title="Third-party (Apache-2.0)", title_zh="第三方（Apache-2.0）", tags=["第三方"],
        desc="Minimal vendored code of CAM++ (3D-Speaker) and HiFT (CosyVoice), each with LICENSE / NOTICE.",
        desc_zh="CAM++（3D-Speaker）与 HiFT（CosyVoice）的最小 vendored 代码，各带 LICENSE / NOTICE。"),
    "MimickingTrain/tests": dict(
        title="pytest", title_zh="pytest", tags=["测试"],
        desc="Parity (our DiT against the reference module, 3e-7), data front end, training-path equivalence, codebook, adversarial head, self-conversion. 28 cases, most of them run on CPU.",
        desc_zh="对拍（我们的 DiT 对参考模块 3e-7）、数据前端、训练路径等价性、码本、对抗头、自转换。28 个用例，CPU 可跑大半。"),
    "MimickingTrain/tools": dict(
        title="Evaluation and diagnostic tools", title_zh="评估与诊断工具", tags=["工具"],
        desc=("stream_diag (streaming block-wise reproduction and localisation, --compute-stats produces the frozen statistics), noise_context (noise "
              "before / after / mixed in), export_parity (ORT against PyTorch), wer_eval, pair_floor (similarity and floor), speaker_probe, "
              "content_gate (codebook fit and threshold), ckpt_index, cv_extract."),
        desc_zh="stream_diag（流式分段复现与定位，--compute-stats 生成冻结统计量）、noise_context（噪声前后 / 混入）、export_parity（ORT 对 PyTorch）、wer_eval、pair_floor（相似度与地板）、speaker_probe、content_gate（码本拟合与门槛）、ckpt_index、cv_extract。"),
    "MimickingTrain/tools/stream_diag.py": dict(
        title="Streaming diagnostics", title_zh="流式诊断", tags=["工具", "正式版"],
        desc="Reproduces the Rust session block by block in PyTorch, matching the Lab's WER; the tool that found the root cause of the 2.0 streaming articulation problem. Use it before exporting.",
        desc_zh="在 PyTorch 里按块复现 Rust 会话，与 Lab 的 WER 吻合；找出 2.0 流式咬字问题根因的工具。导出前先用它。"),
    "MimickingTrain/docs": dict(
        title="Design and records", title_zh="设计与记录", tags=["文档"],
        desc=("despeaker-design (§1–13: despeaker approach, experiments, the 2.0 freeze, Lab measurements, diagnostics), content-despeaker-survey "
              "(survey), lessons-learned (pitfalls), research/ (research notes)."),
        desc_zh="despeaker-design（§1–13：去说话人化方案、实验、2.0 冻结、Lab 实测、诊断）、content-despeaker-survey（调研）、lessons-learned（踩过的坑）、research/（调研笔记）。"),
    "MimickingTrain/docs/lessons-learned.md": dict(
        title="Lessons learned", title_zh="踩过的坑", tags=["文档", "正式版"],
        desc=("Five categories: training-evaluation method, codebook-specific, export, real-time chain, engineering habits; each entry gives "
              "symptom / cause / handling / source; the cleanup record is at the end."),
        desc_zh="训练评估方法、码本特有、导出、实时链路、工程习惯五类，每条症状 / 原因 / 处理 / 出处；末尾是清理记录。"),
    "MimickingTrain/docs/despeaker-design.md": dict(
        title="Design document (main record)", title_zh="设计文档（主记录）", tags=["文档"],
        desc="Numbers and conclusions of all experiments, recorded by section; the official-release material is in §12–13.",
        desc_zh="所有实验的数字与结论按节记录，正式版相关在 §12–13。"),
    "MimickingTrain/runs": dict(
        title="Run records (not in git, cleaned up 2026-11-01)", title_zh="运行记录（不在 git，2026-11-01 已清理）", tags=["记录"],
        desc=("Mimicking-v1.0 / v2.0 are frozen weights; tiny_sp and ft_* are the config, logs and ema_latest of each training run (checkpoints "
              "deleted); eval_*_32 are the 32-pair evaluation outputs (raw data of the three-version table); diag_* / noisectx_* keep only json "
              "and log; lab_* are the Lab measurement records."),
        desc_zh=("Mimicking-v1.0 / v2.0 是冻结权重；tiny_sp、ft_* 是各次训练的配置、日志、ema_latest（检查点已删）；eval_*_32 是 32 对评估输出（三版本表的原始数据）；"
                 "diag_* / noisectx_* 只剩 json 与 log；lab_* 是 Lab 实测记录。"),
        related=["runs/Mimicking-v2.0/RELEASE.md"]),
    "MimickingTrain/runs/Mimicking-v2.0": dict(
        title="2.0 official release weights (frozen)", title_zh="2.0 正式版权重（冻结）", tags=["记录", "正式版", "模型"],
        desc="ema.safetensors, codebook, frozen statistics, config, step_0010000.pt, evaluation json, RELEASE.md (manifest and checksums). Do not modify.",
        desc_zh="ema.safetensors、码本、冻结统计量、config、step_0010000.pt、评估 json、RELEASE.md（清单与校验和）。不要改。"),
    "MimickingTrain/runs/Mimicking-v1.0": dict(
        title="1.0 weights (frozen)", title_zh="1.0 权重（冻结）", tags=["记录", "模型"],
        desc="tiny_sp at step 100 000: ema.safetensors, step_0100000.pt, config, evaluation. The warm-start source of 2.0.",
        desc_zh="tiny_sp 第 100 000 步：ema.safetensors、step_0100000.pt、config、评估。2.0 的热启动来源。"),
    "MimickingTrain/runs/ft_vq4096": dict(
        title="Training run of 2.0", title_zh="2.0 的训练运行", tags=["记录"],
        desc="config, log.jsonl, stdout, synth/ (8-pair listening test every 2 500 steps), tb/, ema_latest (= the EMA at step 10 000). Checkpoints cleaned up.",
        desc_zh="config、log.jsonl、stdout、synth/（每 2 500 步的 8 对试听）、tb/、ema_latest（= 第 10 000 步 EMA）。检查点已清理。"),
    "MimickingTrain/runs/tiny_sp": dict(
        title="Training run of 1.0", title_zh="1.0 的训练运行", tags=["记录"],
        desc="Logs, synth, tb, ema_latest of the 100 k steps; the 32 checkpoints were cleaned up, the final step is in Mimicking-v1.0.",
        desc_zh="100 k 步的日志、synth、tb、ema_latest；32 个检查点已清理，最终步在 Mimicking-v1.0。"),
    # ---------------- MimickingLab
    "MimickingLab": dict(
        title="Desktop test bench (egui, not in git)", title_zh="桌面测试台（egui，不在 git）", tags=["源码"],
        desc=("Real-time microphone conversion, file playback, multi-stream load, quality probe, report export; references Mimicking through a path "
              "dependency. Recommended official-release settings are in its README."),
        desc_zh="实时麦克风转换、文件回放、多路负载、质量探针、报告导出；通过 path 依赖引用 Mimicking。正式版推荐设置见其 README。",
        related=["README.md"]),
    "MimickingLab/README.md": dict(
        title="Usage, recommended settings, metric definitions", title_zh="用法、推荐设置、指标定义", tags=["文档"],
        desc="How to run, headless parameters, 2.0 recommended settings, the meaning of each metric and historical corrections.",
        desc_zh="运行方式、无界面参数、2.0 推荐设置、每个指标的含义与历史修正。"),
    "MimickingLab/crates/mimic-lab/src/main.rs": dict(
        title="UI and headless entry point", title_zh="UI 与无界面入口", tags=["源码"],
        desc=("Settings area (runtime environment, model and block-length radio buttons, export, reference timbre, solver and gating), the real-time / "
              "file / load / quality / report tabs; --source / --live / --export / --gate-probe / --vad / --block etc. for --headless."),
        desc_zh="设置区（运行环境、模型与块长单选、导出、参考音色、求解与门控）、实时 / 文件 / 负载 / 质量 / 报告标签页；--headless 的 --source / --live / --export / --gate-probe / --vad / --block 等。"),
    "MimickingLab/crates/mimic-lab/src/engine.rs": dict(
        title="Engine thread", title_zh="引擎线程", tags=["源码"],
        desc=("Audio streams, file playback, the queue controller (playback-rate steering + backstop pad / drop), metric snapshots, model load / unload, "
              "the export subprocess, the quality-probe thread."),
        desc_zh="音频流、文件回放、队列控制器（播放速率微调 + 兜底补 / 丢）、指标快照、模型加载 / 卸载、导出子进程、质量探针线程。"),
    "MimickingLab/crates/mimic-lab/src/pool.rs": dict(
        title="Per-GPU worker threads", title_zh="每卡工作线程", tags=["源码"],
        desc="One Runner per GPU, streams assigned round-robin by stream number; GateKind (none / energy / Silero) attaches the gate when the session is created.",
        desc_zh="每张卡一个 Runner，路按路号轮转分配；GateKind（无 / 能量 / Silero）在建会话时挂门。"),
    "MimickingLab/crates/mimic-lab/src/audio.rs": dict(
        title="cpal audio streams", title_zh="cpal 音频流", tags=["源码"],
        desc="WASAPI capture / playback; the callback only moves samples (lock-free ring), underrun / glitch counters.",
        desc_zh="WASAPI 采集 / 播放，回调只搬样本（无锁环），欠载 / 断续计数。"),
    "MimickingLab/crates/mimic-lab/src/resample.rs": dict(
        title="Streaming resampling and varispeed stage", title_zh="流式重采样与变速级", tags=["源码"],
        desc="Polyphase sinc resampling (fixed ratio) + Varispeed (±1 % continuous speed change, used by the queue controller).",
        desc_zh="分相 sinc 重采样（固定比例）+ Varispeed（±1 % 连续变速，队列控制器用）。"),
    "MimickingLab/crates/mimic-lab/refs_byname": dict(
        title="Reference timbre library", title_zh="参考音色库", tags=["数据"],
        desc="76 female voices (16 kHz, 13 s) + index.json (name / file / country / age), the source of the Lab's reference drop-down.",
        desc_zh="76 个女声（16 kHz，13 s）+ index.json（name / file / country / age），Lab 的参考下拉列表来源。"),
    "MimickingLab/out": dict(
        title="Lab output (cleared)", title_zh="Lab 输出（已清空）", tags=["构建"],
        desc="<name>_lab.wav from file mode.",
        desc_zh="文件模式的 <名字>_lab.wav。"),
    # ---------------- data root
    "C:/MimickingData": dict(
        title="Data root (in no repository)", title_zh="数据根（不在任何仓库）", tags=["数据"],
        desc="Training and evaluation data: raw corpora, feature cache, codebooks, noise recordings. See the subdirectories.",
        desc_zh="训练与评估的数据：原始语料、特征缓存、码本、噪声录音。见各子目录。"),
    "C:/MimickingData/libritts_r": dict(
        title="LibriTTS-R raw corpus", title_zh="LibriTTS-R 原始语料", tags=["数据", "第三方"],
        desc="The OpenSLR 141 (CC BY 4.0) subsets; the WER evaluation reads its .normalized.txt.",
        desc_zh="OpenSLR 141（CC BY 4.0）各子集；WER 评测读它的 .normalized.txt。"),
    "C:/MimickingData/common-voice": dict(
        title="Common Voice English subset", title_zh="Common Voice 英文子集", tags=["数据", "第三方"],
        desc="CC0; used for data augmentation (ft_cv*).",
        desc_zh="CC0；数据扩充用（ft_cv*）。"),
    "C:/MimickingData/cache": dict(
        title="Feature cache", title_zh="特征缓存", tags=["数据"],
        desc="Shards written by mimictrain.data.cache: int16 waveform + fp16 mel + CAM++ vector, the input of training and evaluation.",
        desc_zh="mimictrain.data.cache 写出的分片：int16 波形 + fp16 梅尔 + CAM++ 向量，训练与评估的输入。"),
    "C:/MimickingData/codebooks": dict(
        title="Content codebooks", title_zh="内容码本", tags=["数据", "正式版"],
        desc="xlsr12_k4096.safetensors (used by 2.0, copied to runs/Mimicking-v2.0) and others.",
        desc_zh="xlsr12_k4096.safetensors（2.0 用，已复制到 runs/Mimicking-v2.0）等。"),
    "C:/MimickingData/noise": dict(
        title="BO's real noise recordings", title_zh="BO 的真实噪声录音", tags=["数据"],
        desc="keyboard / desktop / coughing and sneezing / background noise: the input of the gate evaluation and noise_context.",
        desc_zh="keyboard / desktop / coughing and sneezing / background noise：门控评测与 noise_context 的输入。"),
    "C:/MimickingData/manifests": dict(
        title="Corpus manifests", title_zh="语料清单", tags=["数据"],
        desc="jsonl: path, speaker, duration, text of each utterance.",
        desc_zh="jsonl：每句的路径、说话人、时长、文本。"),
    "C:/MimickingData/audition": dict(
        title="Perturbation listening test", title_zh="扰动试听", tags=["数据"],
        desc="Original / perturbed pairs from mimictrain.data.audition.",
        desc_zh="mimictrain.data.audition 的原始 / 扰动对。"),
}

CATEGORY_RULES = [
    ("Mimicking/models", "模型"), ("Mimicking/ref", "第三方"), ("Mimicking/dist", "构建"),
    ("Mimicking/out", "构建"), ("Mimicking/docs", "文档"), ("Mimicking/tools", "工具"), ("Mimicking/crates", "源码"),
    ("Mimicking/examples", "源码"), ("MimickingTrain/runs", "记录"), ("MimickingTrain/docs", "文档"), ("MimickingTrain/tools", "工具"),
    ("MimickingTrain/tests", "测试"), ("MimickingTrain/configs", "配置"), ("MimickingTrain/mimictrain/thirdparty", "第三方"),
    ("MimickingTrain/mimictrain", "源码"), ("MimickingLab/out", "构建"), ("MimickingLab/crates/mimic-lab/refs_byname", "数据"),
    ("MimickingLab/crates", "源码"), ("C:/MimickingData", "数据"),
]

# ------------------------------------------------------------------ UI strings per language
I18N = {
    "en": {
        "html_lang": "en",
        "title": "Mimicking project tree",
        "sub": ("Every directory and file of the three workspaces (the Mimicking library, MimickingTrain training, the MimickingLab test bench) "
                "and the data root; generated on __DATE__: __NDIRS__ directories, __NFILES__ files. Click an entry for its description; type a "
                "keyword to filter (matches path, title, description, module comment). Generated by <code>tools/project_tree.py</code>; "
                "re-run it when the layout changes."),
        "search": "Search: e.g. manifest, vad, GroupNorm, export, Silero …",
        "expand": "Expand all",
        "collapse": "Collapse to level 2",
        "legend": ("Tags: {tags}; the git marker applies to the Mimicking repository only (tracked, M = modified and not committed, untracked, "
                   "ignored = ignored by .gitignore)."),
        "select_hint": "Select an entry on the left.",
        "switch": 'English | <a href="project-tree.zh-CN.html" lang="zh-CN">中文</a>',
        "zh_version": "Chinese version of {base}",
        "js": {
            "type": "Type", "dir": "directory", "dir_shallow": "directory (listed to this level only)", "file": "file",
            "size": "Size", "files": "Files", "n_files": "{n} files", "git": "git", "related": "Related: ", "sep": ", ",
            "contents": "Contents ({n}):",
            "git_badge": {"tracked": "tracked", "modified": "M", "untracked": "untracked", "ignored": "ignored"},
            "git_state": {"tracked": "tracked", "modified": "modified (not committed)", "untracked": "untracked", "ignored": "ignored (.gitignore)"},
            "units": {"gb": "GB", "mb": "MB", "kb": "KB", "b": "B"},
        },
    },
    "zh-CN": {
        "html_lang": "zh-CN",
        "title": "Mimicking 目录结构",
        "sub": ("三个工作区（Mimicking 库、MimickingTrain 训练、MimickingLab 测试台）与数据根的全部目录与文件；生成于 __DATE__，__NDIRS__ 个目录、__NFILES__ 个文件。"
                "点一项看说明；输入关键字过滤（匹配路径、标题、说明、模块注释）。由 <code>tools/project_tree.py</code> 生成，目录变了重跑即可。"),
        "search": "搜索：如 manifest、vad、GroupNorm、导出、Silero …",
        "expand": "展开全部",
        "collapse": "收起到二级",
        "legend": "标签：{tags}；git 标记只对 Mimicking 仓库有效（tracked = 已跟踪，M = 有改动未提交，untracked = 未跟踪，ignored = 被 .gitignore 忽略）。",
        "select_hint": "在左侧选一项。",
        "switch": '<a href="project-tree.html" lang="en">English</a> | 中文',
        "zh_version": "{base} 的中文版",
        "js": {
            "type": "类型", "dir": "目录", "dir_shallow": "目录（只列到这一级）", "file": "文件",
            "size": "大小", "files": "文件数", "n_files": "{n} 个文件", "git": "git", "related": "相关：", "sep": "、",
            "contents": "内容（{n}）：",
            "git_badge": {"tracked": "tracked", "modified": "M", "untracked": "untracked", "ignored": "ignored"},
            "git_state": {"tracked": "已跟踪", "modified": "有改动未提交", "untracked": "未跟踪", "ignored": "被 .gitignore 忽略"},
            "units": {"gb": "GB", "mb": "MB", "kb": "KB", "b": "B"},
        },
    },
}


def tag_label(key: str, lang: str) -> str:
    return TAG_LABELS.get(key, key) if lang == "en" else key


def category(rel: str) -> str:
    for prefix, cat in CATEGORY_RULES:
        if rel == prefix or rel.startswith(prefix + "/"):
            return cat
    if rel.endswith((".md", ".html")):
        return "文档"
    if rel.endswith((".toml", ".json", ".yml", ".gitignore", ".gitattributes", ".lock")):
        return "配置"
    return "源码"


# a leading language-switch line: `English | [中文](...)` or `[English](...) | 中文` (optionally bold / with trailing spaces)
LANG_SWITCH_LINE = re.compile(r"^\s*\**\s*(?:\[English\]\([^)]*\)|English)\s*\|\s*(?:\[中文\]\([^)]*\)|中文)\s*\**\s*$")
ZH_FILE = re.compile(r"^(.*)\.zh-CN(\.[A-Za-z0-9]+)$")


def doc_line(p: Path) -> str:
    try:
        head = io.open(p, encoding="utf-8", errors="replace").read(4000)
    except Exception:
        return ""
    s = str(p)
    if s.endswith(".rs"):
        m = re.search(r"^//!\s?(.*)$", head, re.M)
    elif s.endswith(".py"):
        m = re.search(r'^\s*r?"""(.*?)(?:"""|\n)', head, re.S)
    elif s.endswith(".md"):
        lines = head.split("\n")
        while lines and (not lines[0].strip() or LANG_SWITCH_LINE.match(lines[0])):
            lines.pop(0)
        m = re.search(r"^#\s+(.*)$", "\n".join(lines), re.M)
    elif s.endswith((".sh", ".toml")):
        m = re.search(r"^#\s?(.*)$", head, re.M)
    else:
        m = None
    return m.group(1).strip().split("\n")[0][:200] if m else ""


def git_states(repo: Path) -> dict[str, str]:
    """relative path -> tracked | modified | untracked | ignored (untracked / ignored entries may be directories)."""
    out: dict[str, str] = {}
    try:
        for line in subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines():
            out[line] = "tracked"
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--ignored"], capture_output=True, text=True, check=True).stdout
        for line in st.splitlines():
            code, path = line[:2], line[3:].strip().rstrip("/")
            if code == "??":
                out[path] = "untracked"
            elif code == "!!":
                out[path] = "ignored"
            elif code.strip():
                out[path] = "modified"
    except Exception as e:  # not a git checkout
        print("git:", e, file=sys.stderr)
    return out


def dir_stats(p: Path) -> tuple[int, int]:
    n = size = 0
    for dp, dns, fns in os.walk(p):
        dns[:] = [d for d in dns if d not in EXCLUDE]
        for f in fns:
            try:
                size += (Path(dp) / f).stat().st_size
            except OSError:
                pass
            n += 1
    return n, size


def git_state_for(rel_in_repo: str, states: dict[str, str]) -> str:
    if rel_in_repo in states:
        return states[rel_in_repo]
    # a file under an untracked / ignored directory inherits it
    parts = rel_in_repo.split("/")
    for i in range(len(parts) - 1, 0, -1):
        s = states.get("/".join(parts[:i]))
        if s in ("untracked", "ignored"):
            return s
    return ""


def zh_sibling_desc(path: str, name: str) -> dict | None:
    """Automatic bilingual description for `X.zh-CN.<ext>`: points at the English sibling `X.<ext>` (path relative to the root)."""
    m = ZH_FILE.match(name)
    if not m:
        return None
    base = m.group(1) + m.group(2)
    root_rel = path.split("/", 1)[1] if "/" in path else path
    related = root_rel[: -len(name)] + base
    return dict(title=I18N["en"]["zh_version"].format(base=base), title_zh=I18N["zh-CN"]["zh_version"].format(base=base),
                desc=I18N["en"]["zh_version"].format(base=base), desc_zh=I18N["zh-CN"]["zh_version"].format(base=base),
                tags=["文档"], related=[related])


def scan() -> list[dict]:
    nodes: list[dict] = []
    git = git_states(HERE)
    for root_name, root in ROOTS:
        if not root.exists():
            continue
        for dp, dns, fns in os.walk(root):
            rel = root_name + ("/" + Path(dp).relative_to(root).as_posix() if Path(dp) != root else "")
            dns[:] = sorted(d for d in dns if d not in EXCLUDE)
            n, size = dir_stats(Path(dp))
            node = dict(path=rel, name=Path(dp).name if Path(dp) != root else root_name, kind="dir", size=size, files=n,
                        children=sorted(dns) + sorted(fns), cat=category(rel))
            if root_name == "Mimicking":
                node["git"] = git_state_for(Path(dp).relative_to(root).as_posix(), git) if Path(dp) != root else "tracked"
            nodes.append(node)
            shallow = rel in SHALLOW
            if shallow:
                for d in dns:
                    sub = Path(dp) / d
                    sn, ss = dir_stats(sub)
                    nodes.append(dict(path=f"{rel}/{d}", name=d, kind="dir", size=ss, files=sn, children=[], summary=True, cat=category(f"{rel}/{d}"),
                                      git=git_state_for((sub.relative_to(root)).as_posix(), git) if root_name == "Mimicking" else None))
                dns[:] = []
            for f in sorted(fns):
                fp = Path(dp) / f
                try:
                    sz = fp.stat().st_size
                except OSError:
                    sz = 0
                entry = dict(path=f"{rel}/{f}", name=f, kind="file", size=sz, doc=doc_line(fp) if not shallow else "", cat=category(f"{rel}/{f}"))
                if root_name == "Mimicking":
                    entry["git"] = git_state_for(fp.relative_to(root).as_posix(), git)
                nodes.append(entry)
    # data root: one level, names only
    if DATA_ROOT.exists():
        kids = sorted(p.name for p in DATA_ROOT.iterdir())
        nodes.append(dict(path="C:/MimickingData", name="C:/MimickingData", kind="dir", size=0, files=0, children=kids, cat="数据", external=True))
        for k in kids:
            p = DATA_ROOT / k
            nodes.append(dict(path=f"C:/MimickingData/{k}", name=k, kind="dir" if p.is_dir() else "file", size=p.stat().st_size if p.is_file() else 0,
                              files=0, children=[], summary=True, cat="数据", external=True))
    for nd in nodes:
        d = DESC.get(nd["path"])
        if d is None and nd["kind"] == "file":
            d = zh_sibling_desc(nd["path"], nd["name"])
        if d:
            nd.update(title=d.get("title", ""), title_zh=d.get("title_zh", ""), desc=d.get("desc", ""), desc_zh=d.get("desc_zh", ""),
                      tags=d.get("tags", []), related=d.get("related", []))
    return nodes


def localise(nodes: list[dict], lang: str) -> list[dict]:
    """Keep only the title / desc of one language (the JS reads n.title and n.desc)."""
    out = []
    for n in nodes:
        m = dict(n)
        zt, zd = m.pop("title_zh", None), m.pop("desc_zh", None)
        if lang == "zh-CN":
            if zt is not None:
                m["title"] = zt
            if zd is not None:
                m["desc"] = zd
        out.append(m)
    return out


def render(nodes: list[dict], lang: str) -> str:
    L = I18N[lang]
    data = json.dumps(localise(nodes, lang), ensure_ascii=False)
    tags = {k: tag_label(k, lang) for k in TAG_LABELS}
    gen = dt.date.today().isoformat()
    n_files = sum(1 for n in nodes if n["kind"] == "file")
    n_dirs = sum(1 for n in nodes if n["kind"] == "dir")
    page = TEMPLATE
    for k, v in {
        "__HTML_LANG__": L["html_lang"], "__TITLE__": html.escape(L["title"]), "__SUB__": L["sub"], "__SEARCH__": html.escape(L["search"]),
        "__EXPAND__": html.escape(L["expand"]), "__COLLAPSE__": html.escape(L["collapse"]),
        "__LEGEND__": html.escape(L["legend"].format(tags=" / ".join(tags[c] for c in CATS))),
        "__SELECT_HINT__": html.escape(L["select_hint"]), "__SWITCH__": L["switch"],
        "__I18N__": json.dumps(L["js"], ensure_ascii=False), "__TAGS__": json.dumps(tags, ensure_ascii=False),
        "__CATS__": json.dumps(CATS, ensure_ascii=False), "__DATA__": data,
    }.items():
        page = page.replace(k, v)
    return page.replace("__DATE__", gen).replace("__NFILES__", str(n_files)).replace("__NDIRS__", str(n_dirs))


TEMPLATE = r"""<!doctype html>
<html lang="__HTML_LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;800&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg:#f5f6f4; --panel:#ffffff; --fg:#1b2420; --muted:#5d6a64; --line:#d6dcd8; --accent:#0e6f6b; --accent-soft:#d9ecea; --hot:#c2552a;
  --c-源码:#0e6f6b; --c-工具:#7a4fb0; --c-文档:#2f6db5; --c-模型:#b06a1a; --c-数据:#5b7d2a; --c-记录:#8a7a2a; --c-第三方:#8a3f6e; --c-构建:#777; --c-配置:#4b6a8a; --c-测试:#2a7a6a;
  --font-display:"Manrope","Segoe UI",system-ui,sans-serif; --font-mono:"IBM Plex Mono",Consolas,"Courier New",monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121715; --panel:#1a211e; --fg:#e6ebe8; --muted:#9aa8a1; --line:#2d3732; --accent:#5fc3bd; --accent-soft:#19403d; --hot:#e8865e; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#121715; --panel:#1a211e; --fg:#e6ebe8; --muted:#9aa8a1; --line:#2d3732; --accent:#5fc3bd; --accent-soft:#19403d; --hot:#e8865e; color-scheme:dark; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 var(--font-display); }
header { padding:16px 20px 10px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:5; }
header h1 { margin:0 0 4px; font-size:20px; font-weight:800; }
header h1 .lang { float:right; font-size:13px; font-weight:500; color:var(--muted); }
header h1 .lang a { color:var(--accent); text-decoration:none; }
header h1 .lang a:hover { text-decoration:underline; }
header .sub { color:var(--muted); font-size:13px; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
.bar input { flex:1 1 260px; min-width:200px; padding:7px 10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); font:14px var(--font-display); }
.chip { border:1px solid var(--line); border-radius:999px; padding:3px 10px; font-size:12px; cursor:pointer; background:var(--panel); color:var(--fg); user-select:none; }
.chip.on { background:var(--accent-soft); border-color:var(--accent); }
.chip .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
button.sm { border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--fg); padding:5px 10px; font-size:12px; cursor:pointer; }
main { display:grid; grid-template-columns: minmax(360px, 1.3fr) minmax(300px, 1fr); gap:0; height:calc(100vh - 118px); }
#tree { overflow:auto; padding:10px 8px 40px 12px; border-right:1px solid var(--line); }
#detail { overflow:auto; padding:16px 20px 40px; }
.node { display:flex; align-items:baseline; gap:6px; padding:2px 6px; border-radius:6px; cursor:pointer; white-space:nowrap; }
.node:hover { background:var(--accent-soft); }
.node.sel { background:var(--accent-soft); outline:1px solid var(--accent); }
.node .tw { width:14px; color:var(--muted); font-size:11px; flex:none; text-align:center; }
.node .nm { font-family:var(--font-mono); font-size:13px; }
.node .nm.dir { font-weight:600; }
.node .ti { color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; }
.node .sz { margin-left:auto; color:var(--muted); font-size:11px; font-family:var(--font-mono); flex:none; padding-left:10px; }
.badge { display:inline-block; font-size:10px; padding:0 6px; border-radius:999px; color:#fff; line-height:16px; flex:none; }
.gbadge { font-size:10px; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:0 4px; line-height:14px; flex:none; }
.gbadge.modified { color:var(--hot); border-color:var(--hot); }
.kids { margin-left:16px; border-left:1px dashed var(--line); padding-left:4px; }
.hidden { display:none; }
mark { background:#ffe58a; color:#000; border-radius:2px; padding:0 1px; }
#detail h2 { margin:0 0 2px; font-size:18px; }
#detail .path { font-family:var(--font-mono); font-size:12px; color:var(--muted); word-break:break-all; }
#detail .meta { display:grid; grid-template-columns:auto 1fr; gap:3px 12px; margin:12px 0; font-size:13px; }
#detail .meta dt { color:var(--muted); }
#detail .meta dd { margin:0; }
#detail .desc { margin:10px 0; }
#detail .doc { font-family:var(--font-mono); font-size:12px; color:var(--muted); background:var(--bg); padding:8px 10px; border-radius:6px; border:1px solid var(--line); }
#detail ul.kids { list-style:none; margin:6px 0; padding:0 0 0 4px; border:0; }
#detail ul.kids li { font-family:var(--font-mono); font-size:12px; cursor:pointer; padding:1px 0; }
#detail ul.kids li:hover { color:var(--accent); }
.legend { font-size:12px; color:var(--muted); margin-top:6px; }
@media (max-width: 860px) { main { grid-template-columns:1fr; height:auto; } #tree { border-right:0; border-bottom:1px solid var(--line); max-height:55vh; } }
</style>
</head>
<body>
<header>
  <h1>__TITLE__ <span class="lang">__SWITCH__</span></h1>
  <div class="sub">__SUB__</div>
  <div class="bar">
    <input id="q" type="search" placeholder="__SEARCH__" autocomplete="off">
    <span id="chips"></span>
    <button class="sm" id="expand">__EXPAND__</button>
    <button class="sm" id="collapse">__COLLAPSE__</button>
  </div>
  <div class="legend">__LEGEND__</div>
</header>
<main>
  <div id="tree"></div>
  <div id="detail"><p class="legend">__SELECT_HINT__</p></div>
</main>
<script>
const NODES = __DATA__;
const L = __I18N__;
const TAG = __TAGS__;
const byPath = new Map(NODES.map(n => [n.path, n]));
const CATS = __CATS__;
const COLORS = {源码:"#0e6f6b",工具:"#7a4fb0",文档:"#2f6db5",模型:"#b06a1a",数据:"#5b7d2a",记录:"#8a7a2a",第三方:"#8a3f6e",构建:"#777777",配置:"#4b6a8a",测试:"#2a7a6a"};
const active = new Set(CATS);
const U = L.units;
const fmt = b => b >= 1<<30 ? (b/(1<<30)).toFixed(1)+" "+U.gb : b >= 1<<20 ? (b/(1<<20)).toFixed(1)+" "+U.mb : b >= 1024 ? (b/1024).toFixed(0)+" "+U.kb : b ? b+" "+U.b : "";
const tag = t => TAG[t] || t;
const nFiles = n => L.n_files.replace("{n}", n);
const roots = ["Mimicking","MimickingTrain","MimickingLab","C:/MimickingData"];
const childrenOf = p => { const n = byPath.get(p); if (!n || n.kind !== "dir") return []; return (n.children||[]).map(c => byPath.get(p + "/" + c)).filter(Boolean); };
const esc = s => (s||"").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const hl = (s, q) => { s = esc(s); if (!q) return s; const i = s.toLowerCase().indexOf(q.toLowerCase()); return i < 0 ? s : s.slice(0,i) + "<mark>" + s.slice(i, i+q.length) + "</mark>" + s.slice(i+q.length); };
const open = new Set(roots);
let selected = null, query = "";
function matches(n, q) { if (!q) return true; q = q.toLowerCase(); return [n.path, n.title, n.desc, n.doc, (n.tags||[]).join(" "), (n.tags||[]).map(tag).join(" ")].some(s => (s||"").toLowerCase().includes(q)); }
function subtreeMatches(n, q) { if (matches(n, q)) return true; return childrenOf(n.path).some(c => subtreeMatches(c, q)); }
function render() {
  const tree = document.getElementById("tree"); tree.innerHTML = "";
  for (const r of roots) { const n = byPath.get(r); if (n) tree.appendChild(renderNode(n, 0)); }
}
function renderNode(n, depth) {
  const wrap = document.createElement("div");
  const vis = active.has(n.cat) || n.kind === "dir";
  if (!vis || (query && !subtreeMatches(n, query))) { wrap.className = "hidden"; return wrap; }
  const row = document.createElement("div"); row.className = "node" + (selected === n.path ? " sel" : "");
  const kids = childrenOf(n.path); const isOpen = open.has(n.path) || (query && subtreeMatches(n, query) && !matches(n, query)) || (query && kids.some(c => subtreeMatches(c, query)));
  const tw = document.createElement("span"); tw.className = "tw"; tw.textContent = n.kind === "dir" ? (n.summary ? "▪" : (isOpen ? "▾" : "▸")) : "·";
  const nm = document.createElement("span"); nm.className = "nm" + (n.kind === "dir" ? " dir" : ""); nm.innerHTML = hl(n.name + (n.kind === "dir" ? "/" : ""), query);
  row.append(tw, nm);
  const b = document.createElement("span"); b.className = "badge"; b.style.background = COLORS[n.cat] || "#777"; b.textContent = tag(n.cat); row.appendChild(b);
  if (n.git) { const g = document.createElement("span"); g.className = "gbadge " + n.git; g.textContent = L.git_badge[n.git] || n.git; row.appendChild(g); }
  const ti = document.createElement("span"); ti.className = "ti"; ti.innerHTML = hl(n.title || n.doc || (n.summary ? nFiles(n.files) : ""), query); row.appendChild(ti);
  const sz = document.createElement("span"); sz.className = "sz"; sz.textContent = fmt(n.size); row.appendChild(sz);
  row.onclick = e => { select(n.path); if (n.kind === "dir" && !n.summary) { if (open.has(n.path)) open.delete(n.path); else open.add(n.path); render(); } };
  wrap.appendChild(row);
  if (n.kind === "dir" && isOpen && !n.summary) { const k = document.createElement("div"); k.className = "kids"; for (const c of kids) k.appendChild(renderNode(c, depth + 1)); wrap.appendChild(k); }
  return wrap;
}
function select(p) {
  selected = p; const n = byPath.get(p); if (!n) return;
  location.hash = encodeURIComponent(p);
  const d = document.getElementById("detail");
  const kids = childrenOf(p);
  d.innerHTML = `<h2>${esc(n.title || n.name)}</h2><div class="path">${esc(n.path)}</div>
    <dl class="meta">
      <dt>${L.type}</dt><dd>${n.kind === "dir" ? (n.summary ? L.dir_shallow : L.dir) : L.file}　<span class="badge" style="background:${COLORS[n.cat]||"#777"}">${esc(tag(n.cat))}</span>${(n.tags||[]).filter(t=>t!==n.cat).map(t=>` <span class="badge" style="background:${COLORS[t]||"#555"}">${esc(tag(t))}</span>`).join("")}</dd>
      ${n.size ? `<dt>${L.size}</dt><dd>${fmt(n.size)}${n.kind === "dir" ? `${L.sep}${nFiles(n.files)}` : ""}</dd>` : (n.kind === "dir" && n.files ? `<dt>${L.files}</dt><dd>${n.files}</dd>` : "")}
      ${n.git ? `<dt>${L.git}</dt><dd>${esc(L.git_state[n.git] || n.git)}</dd>` : ""}
    </dl>
    ${n.desc ? `<div class="desc">${esc(n.desc)}</div>` : ""}
    ${n.doc ? `<div class="doc">${esc(n.doc)}</div>` : ""}
    ${(n.related||[]).length ? `<p class="legend">${L.related}${n.related.map(r => `<code>${esc(r)}</code>`).join(L.sep)}</p>` : ""}
    ${kids.length ? `<p class="legend">${L.contents.replace("{n}", kids.length)}</p><ul class="kids">${kids.map(c => `<li data-p="${esc(c.path)}">${c.kind === "dir" ? "▸ " : "· "}${esc(c.name)}${c.kind === "dir" ? "/" : ""}${c.title || c.doc ? ` <span class="legend">— ${esc((c.title || c.doc).slice(0, 80))}</span>` : ""}</li>`).join("")}</ul>` : ""}`;
  d.querySelectorAll("li[data-p]").forEach(li => li.onclick = () => { const q = li.dataset.p; let a = q; while (a.includes("/")) { a = a.slice(0, a.lastIndexOf("/")); open.add(a); } open.add(q); select(q); render(); });
  render();
}
document.getElementById("q").addEventListener("input", e => { query = e.target.value.trim(); render(); });
document.getElementById("expand").onclick = () => { NODES.filter(n => n.kind === "dir" && !n.summary).forEach(n => open.add(n.path)); render(); };
document.getElementById("collapse").onclick = () => { open.clear(); roots.forEach(r => open.add(r)); NODES.filter(n => n.kind === "dir" && n.path.split("/").length === 2 && !n.path.startsWith("C:")).forEach(n => open.add(n.path)); render(); };
const chips = document.getElementById("chips");
for (const c of CATS) { const s = document.createElement("span"); s.className = "chip on"; s.innerHTML = `<span class="dot" style="background:${COLORS[c]}"></span>${esc(tag(c))}`; s.onclick = () => { if (active.has(c)) active.delete(c); else active.add(c); s.classList.toggle("on"); render(); }; chips.appendChild(s); }
document.getElementById("collapse").onclick();
if (location.hash.length > 1) { const p = decodeURIComponent(location.hash.slice(1)); if (byPath.has(p)) { let a = p; while (a.includes("/")) { a = a.slice(0, a.lastIndexOf("/")); open.add(a); } select(p); } }
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the bilingual project-tree pages.")
    ap.add_argument("--lang", choices=[*LANGS, "all"], default="all", help="which page to write (default: all)")
    args = ap.parse_args()
    langs = LANGS if args.lang == "all" else (args.lang,)
    nodes = scan()
    for lang in langs:
        out = OUT[lang]
        out.write_text(render(nodes, lang), encoding="utf-8", newline="\n")
        print(f"wrote {out} ({len(nodes)} entries, {lang})")
    missing = [k for k in DESC if k not in {n["path"] for n in nodes}]
    if missing:
        print("DESC entries with no matching path:", missing)


if __name__ == "__main__":
    main()
