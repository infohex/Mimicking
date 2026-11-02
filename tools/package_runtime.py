"""Assemble a self-contained runtime folder (the layout `mimic` and the library discover by default):

    <dest>/
      mimic.exe                      (from target/release, if built)
      runtime/onnxruntime/           onnxruntime.dll, onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll
      runtime/cuda/                  cudart, cublas, cublasLt, cudnn*, cufft, curand, nvrtc  (what the CUDA EP loads)
      models/<set>/                  one model set: graphs + manifest.json (default Mimicking-v2.0-lowlat)
      RUNTIME-LAYOUT.md

Sources default to the project's .venv (onnxruntime-gpu 1.22 + torch's bundled CUDA 12 libraries).

    python tools/package_runtime.py --dest dist/mimic-win64 [--venv .venv] [--set Mimicking-v2.0-lowlat | --no-models]
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORT_DLLS = ["onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"]
# cuDNN 9 is split into many DLLs; cuBLAS needs cublasLt; cufft/curand/nvrtc are pulled in by some kernels
CUDA_PREFIXES = ["cudart64_", "cublas64_", "cublasLt64_", "cudnn64_", "cudnn_", "cufft64_", "curand64_", "nvrtc64_", "nvrtc-builtins64_", "cupti64_"]

LAYOUT_DOC = """# Runtime layout

`mimic` (and `mimic_onnx::OnnxRuntime::discover`) look for ONNX Runtime in this order:

1. `--ort <path>` / an explicit path given in code
2. `$MIMIC_ORT_DIR/onnxruntime.dll`
3. `<exe dir>/runtime/onnxruntime/onnxruntime.dll`
4. `$ORT_DYLIB_PATH`

The execution-provider libraries (`onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll`) must sit
next to `onnxruntime.dll`. Before loading, `<exe dir>/runtime/cuda` and `$MIMIC_CUDA_DIR` are prepended to `PATH`
so the CUDA EP finds cudart / cuBLAS / cuDNN there; a system-wide CUDA 12 + cuDNN 9 install works too.

Models: `--models <dir>` (default: the first set with a manifest under `<exe dir>/models`, then `./models`, `*-lowlat` preferred). `mimic models verify` checks the
files against `manifest.json`.

Versions this folder was assembled with are listed in VERSIONS.txt.
"""


def copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return src.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--venv", default=str(ROOT / ".venv"))
    ap.add_argument("--set", default="Mimicking-v2.0-lowlat", help="model set under models/ to package (copied under the same name)")
    ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args()
    venv = Path(a.venv)
    sp = venv / "Lib" / "site-packages"
    dest = Path(a.dest)
    total = 0
    versions = []

    ort_dir = sp / "onnxruntime" / "capi"
    for n in ORT_DLLS:
        p = ort_dir / n
        if not p.exists():
            sys.exit(f"missing {p}")
        total += copy(p, dest / "runtime" / "onnxruntime")
    try:
        versions.append("onnxruntime " + (sp / "onnxruntime" / "capi" / "version_info.py").read_text().split("\n")[0])
    except Exception:  # noqa: BLE001
        pass
    import importlib.metadata as md
    for pkg in ("onnxruntime-gpu", "torch"):
        try:
            versions.append(f"{pkg} {md.version(pkg)}")
        except Exception:  # noqa: BLE001
            pass

    cuda_src = sp / "torch" / "lib"
    n_cuda = 0
    for p in sorted(cuda_src.glob("*.dll")):
        if any(p.name.startswith(pre) for pre in CUDA_PREFIXES):
            total += copy(p, dest / "runtime" / "cuda")
            n_cuda += 1
    if n_cuda == 0:
        sys.exit(f"no CUDA libraries found under {cuda_src}")

    exe = ROOT / "target" / "release" / "mimic.exe"
    if exe.exists():
        total += copy(exe, dest)
    if not a.no_models:
        mdir = ROOT / "models" / a.set
        if not (mdir / "manifest.json").is_file():
            sys.exit(f"{mdir} has no manifest.json (pass --set <name> or --no-models)")
        for p in sorted(mdir.iterdir()):
            if p.is_file() and p.suffix in (".onnx", ".npz", ".json"):
                total += copy(p, dest / "models" / a.set)
    (dest / "RUNTIME-LAYOUT.md").write_text(LAYOUT_DOC)
    (dest / "VERSIONS.txt").write_text("\n".join(versions) + "\n")
    print(f"assembled {dest} ({total / 1e6:.0f} MB; ORT {len(ORT_DLLS)} files, CUDA {n_cuda} files)")


if __name__ == "__main__":
    main()
