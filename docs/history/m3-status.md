English | [中文](m3-status.zh-CN.md)

# M3 status: preparing the library for release (2026-10-26)

## Conclusion

**M3 complete.** The acceptance gate "an external crate depends only on `mimic-pipeline` and `mimic-model` and writes a conversion program in under 50 lines" is met: the body of `examples/convert-minimal` is 40 lines (including wav read/write). `cargo clippy -D warnings`, `cargo doc -D warnings` and all tests pass; the self-contained runtime directory runs in a clean environment; the first git commit is done (local, not pushed).

## Deliverables

| Item | Location | Notes |
|---|---|---|
| API entry points | re-exported `mimic_model::{OnnxRuntime, Device, ModelSet::into_stages, geometry}`; `mimic_pipeline::{Runner, Session, Converter, EulerCfg}` | The integrating application does not need to depend directly on `mimic-onnx` or `mimic-core` |
| Example | `examples/convert-minimal` | `cargo run --release -p convert-minimal -- ref.wav src.wav out.wav` |
| Documentation | `README.md`, module-level rustdoc for each crate, `docs/*.md` | `cargo doc --workspace --no-deps` with zero warnings |
| Model manifest | `models/onnx/manifest.json` (`tools/export/make_manifest.py`) | 10 files, sha256, size, role, export configuration, weight provenance and licence |
| Manifest verification | `ModelFiles::verify()`, `mimic models verify` / `models list` | 10 files, 819 MB verified in 0.8 s |
| Runtime library directory convention | `OnnxRuntime::discover()`, `RUNTIME-LAYOUT.md` | Search order: explicit path → `$MIMIC_ORT_DIR` → `<exe>/runtime/onnxruntime` → `$ORT_DYLIB_PATH`; `<exe>/runtime/cuda` and `$MIMIC_CUDA_DIR` are added to PATH |
| Packaging script | `tools/package_runtime.py --dest dist/mimic-win64` | Collects the 3 ORT DLLs and 16 CUDA DLLs from `.venv`, adds the exe and the models; 3.0 GB |
| CI | `.github/workflows/ci.yml` | Windows + Ubuntu: fmt, clippy -D warnings, build, non-GPU tests, doc; a manually triggered self-hosted GPU job runs the manifest verification and all parity tests |
| Changelog | `CHANGELOG.md` 0.1.0 | |

## Verification

- `cargo test --workspace --release`: the six dsp parity tests, SOLA and other unit tests pass; the ONNX tests skip automatically when `MIMIC_ORT_DYLIB` is not set.
- GPU parity (`MIMIC_ORT_DYLIB`, `MIMIC_GPU=1`): 7/7 pass.
- Clean environment (`env -i`, only the system PATH) running `dist/mimic-win64/mimic.exe`: `models verify` passes, `convert` per block p50 48.0 / p95 52.4 ms.

## Licence and distribution boundary

- The model card of the original seed-vc weights `Plachta/Seed-VC` declares **GPL-3.0**; XLS-R and CosyVoice HiFT are Apache-2.0; CAM++ is released with FunASR. The exported graphs are derivatives of these weights; **this repository distributes neither the weights nor the graphs**, users build them themselves with `tools/export`; the manifest records provenance and licence.
- The code in this repository contains no original seed-vc code; the export scripts in `tools/export` take the original seed-vc repository as a runtime dependency (`ref/seed-vc` is not checked in). The code licence was provisionally MIT OR Apache-2.0 at the time; it was finalised as Apache-2.0 on 2026-11-01.

## Known items and follow-ups

- `xlsr.onnx` and `xlsr_ref.onnx` each carry a 329 MB copy of the weights; switching to a shared external weight file would save 329 MB in the model directory.
- Of the 3.0 GB of runtime libraries, cuDNN's precompiled engines are the bulk; if only the CUDA EP's conv search results are used, the optional parts other than `cudnn_engines_precompiled64_9.dll` could be tried without, which needs measuring.
- The CI GPU job needs a self-hosted runner with the `gpu` label and two repository variables (`ORT_DYLIB_PATH`, `MIMIC_GPU`).
- Configuration changes such as block length require re-exporting a set of graphs into another directory; the `config` field of `manifest.json` identifies them.
