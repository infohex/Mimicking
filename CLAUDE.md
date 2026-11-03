# Mimicking — 项目规则

Rust 实时零样本变声库，库优先、模块化（每级一个 trait）。2.0 起生成器（调节器 + DiT）为 MimickingTrain 自训权重；链路结构沿袭 seed-vc tiny，其原版权重只作评测基线与导出共享图（`onnx*` 基线套）。文档里称它"基线"，不写"上游"。
核心 crate 不包含任何使用方的知识（采样率适配、管道协议、配置文件都属于使用方）。

## 运行时规则（改任何涉及 ORT / CUDA / 多线程的代码前先读 docs/runtime-pitfalls.md）

1. 进程内只有一个 ONNX Runtime 环境：永远复用 `OnnxRuntime`，换设备用 `with_device`，不要第二次 `load`。
2. CUDA Graph 录制（`BoundGraph::load` / `warm_up`）必须在所有实时推理开始之前完成，且不能和其他线程的 CUDA 调用并发；
   库里用 `CAPTURE_LOCK` 串行化加载，使用方要保证"先全部加载、再开始跑"。
3. 一张卡一个线程一套 `Runner`；`BoundGraph` 不跨线程共用；多卡 = 多个 Runner 并行，不是一个线程轮询。
4. ORT 线程池不自旋（已在 `OnnxGraph::load` 设置），不要去掉。
5. 图是静态形状：几何只从模型目录的 manifest 读，不在代码里写死；改配置 = 重新导出一套到新目录。
6. 慢活（嵌入、IO、子进程）不进推理线程。
7. VAD 决策按算法延迟对齐到输出块（`Session::roll` 跨过的静音块同样把决策入队）。
8. 码本模型（2.0 起）的导出必须带 `--norm-stats`（调节器 GroupNorm 统计量冻成整句常数）；导出前先用 MimickingTrain
   `tools/stream_diag.py --segmented` 复现流式 WER，离线 WER 不代表流式。自训套整套由 `mimictrain.export` 导出（XLS-R / CAM++ / HiFT
   也从 `ref/models` 的 Apache 权重导，不经 `tools/export`），所有图的输入输出必须是 fp32（运行时按声明类型绑定缓冲）。
9. 自训模型套命名 `Mimicking-v<主.次>`（目录、manifest `name`），图版本号与权重版本号分开；基线套保持 `onnx*`。
10. 语音门默认 `SileroVad`（`models/vad/silero_vad.onnx`）；能量门只作退路；离线文件转换不挂门。
11. 使用方的扬声器队列用播放速率微调（±1 %）吸收时钟漂移，补静音 / 丢输入只兜底；单路评估延迟时先锁 GPU 频率。

## 工程约定

- 对拍优先：改任何一级的数值路径，先跑 `cargo test --release`（需 `MIMIC_ORT_DYLIB`、`MIMIC_GPU`，见 README），
  相对 L2 要在 fp16 噪声内。
- 自训套的导出在 MimickingTrain（`python -m mimictrain.export`，夹具写到套目录的 `fixtures/`）；基线导出链 `tools/export`（import seed-vc，GPL）已于 2026-11-01 连同 `ref/seed-vc`、`bench/` 归档到 `C:/Workspace/Rust/_archive/Mimicking-seedvc-2026-11-01`，
  基线套只保留已导出的 `models/onnx*`。Rust 对拍测试用
  `MIMIC_MODELS=<套目录>`（夹具取 `<套>/fixtures`，或 `MIMIC_FIXTURES`），CUDA / cuDNN DLL 目录要在 PATH 上。
- 库代码许可 Apache-2.0（`LICENSE` + `NOTICE`，2026-11-01 定稿）；基线权重 GPL-3.0，仓库不分发权重或导出的图。
- 文档双语：英文为默认版本（原路径，如 `README.md`、`docs/release-2.0.md`），中文为同名 `.zh-CN` 文件（`README.zh-CN.md`），每个文件顶部一行语言切换。
  改任一文档必须同步另一语言版本，三个工作区（Mimicking / MimickingTrain / MimickingLab）的 README 与 docs 都适用；中文版内部链接指向 `.zh-CN` 同级文件。
  `docs/project-tree.html` 与 `.zh-CN.html` 由 `tools/project_tree.py` 一次生成两份（说明表 `DESC` 双语），不手改。本文件（CLAUDE.md）与 `docs/research/` 只保留原语言。
- 测试台在独立 workspace `../MimickingLab`，通过 path 依赖引用本库；库的依赖树里不放 GUI/音频设备相关的东西。
