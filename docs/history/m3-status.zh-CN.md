[English](m3-status.md) | 中文

# M3 状态：库发布准备（2026-10-26）

## 结论

**M3 完成。** 验收门"外部 crate 只依赖 `mimic-pipeline` 和 `mimic-model`，50 行以内写出转换程序"达成：`examples/convert-minimal` 主体 40 行（含 wav 读写）。`cargo clippy -D warnings`、`cargo doc -D warnings`、全部测试通过；自包含运行目录在干净环境下可运行；首次 git 提交已完成（本地，未推送）。

## 交付物

| 项 | 位置 | 说明 |
|---|---|---|
| API 入口 | `mimic_model::{OnnxRuntime, Device, ModelSet::into_stages, geometry}` 再导出；`mimic_pipeline::{Runner, Session, Converter, EulerCfg}` | 使用方不需要直接依赖 `mimic-onnx` 或 `mimic-core` |
| 示例 | `examples/convert-minimal` | `cargo run --release -p convert-minimal -- ref.wav src.wav out.wav` |
| 文档 | `README.md`、各 crate 的模块级 rustdoc、`docs/*.md` | `cargo doc --workspace --no-deps` 零警告 |
| 模型清单 | `models/onnx/manifest.json`（`tools/export/make_manifest.py`） | 10 个文件、sha256、大小、角色、导出配置、权重来源与许可 |
| 清单校验 | `ModelFiles::verify()`、`mimic models verify` / `models list` | 10 个文件 819 MB 校验 0.8 s |
| 运行库目录约定 | `OnnxRuntime::discover()`、`RUNTIME-LAYOUT.md` | 查找顺序：显式路径 → `$MIMIC_ORT_DIR` → `<exe>/runtime/onnxruntime` → `$ORT_DYLIB_PATH`；`<exe>/runtime/cuda`、`$MIMIC_CUDA_DIR` 加到 PATH |
| 打包脚本 | `tools/package_runtime.py --dest dist/mimic-win64` | 从 `.venv` 收集 ORT 3 个 DLL 与 CUDA 16 个 DLL，加 exe 与模型；3.0 GB |
| CI | `.github/workflows/ci.yml` | Windows + Ubuntu：fmt、clippy -D warnings、build、无 GPU 测试、doc；手动触发的 self-hosted GPU 作业跑清单校验与全部对拍 |
| 变更日志 | `CHANGELOG.md` 0.1.0 | |

## 验证

- `cargo test --workspace --release`：dsp 六项对拍、SOLA 等单元测试通过；ONNX 测试在没有 `MIMIC_ORT_DYLIB` 时自动跳过。
- GPU 对拍（`MIMIC_ORT_DYLIB`、`MIMIC_GPU=1`）：7/7 通过。
- 干净环境（`env -i`，只有系统 PATH）运行 `dist/mimic-win64/mimic.exe`：`models verify` 通过，`convert` 每块 p50 48.0 / p95 52.4 ms。

## 许可与分发边界

- 基线权重 `Plachta/Seed-VC` 的模型卡声明 **GPL-3.0**；XLS-R 与 CosyVoice HiFT 为 Apache-2.0；CAM++ 以 FunASR 发布。导出的图是这些权重的派生物，**本仓库不分发权重和图**，由使用者用 `tools/export` 自建；manifest 记录来源与许可。
- 本仓库代码不含 seed-vc 原版代码；`tools/export` 的导出脚本以原版仓库为运行时依赖（`ref/seed-vc` 不入库）。代码许可当时暂写 MIT OR Apache-2.0，2026-11-01 定为 Apache-2.0。

## 已知事项与后续

- `xlsr.onnx` 与 `xlsr_ref.onnx` 各带一份 329 MB 权重；可改为外部权重文件共享，模型目录能省 329 MB。
- 运行库 3.0 GB 里 cuDNN 的预编译引擎占大头；若只用 CUDA EP 的 conv 搜索结果，可试去掉 `cudnn_engines_precompiled64_9.dll` 以外的可选项，需要实测。
- CI 的 GPU 作业需要一台带 `gpu` 标签的 self-hosted runner 和两个仓库变量（`ORT_DYLIB_PATH`、`MIMIC_GPU`）。
- 块长等配置变更需要重新导出一套图到另一目录；`manifest.json` 的 `config` 字段用于识别。
