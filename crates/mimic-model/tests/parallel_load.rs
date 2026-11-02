//! Loading full model sets on several devices from several threads at once must work: CUDA Graph capture is
//! serialised inside `mimic-onnx` (`CAPTURE_LOCK`), and the ORT environment is created once per process.
//! Needs `MIMIC_ORT_DYLIB`, the models directory, and `MIMIC_GPUS=0,1` (two or more CUDA ordinals); skipped
//! otherwise.

use mimic_model::{Device, ModelFiles, OnnxRuntime, ModelSet};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    std::env::var_os("MIMIC_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Mimicking-v2.0-lowlat"))
}

#[test]
fn parallel_loads_on_several_gpus_do_not_break_capture() {
    let Some(gpus) = std::env::var("MIMIC_GPUS").ok() else {
        eprintln!("MIMIC_GPUS not set: skipped");
        return;
    };
    let gpus: Vec<i32> = gpus.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if gpus.len() < 2 {
        eprintln!("MIMIC_GPUS needs two or more ordinals: skipped");
        return;
    }
    let dylib = std::env::var_os("MIMIC_ORT_DYLIB").expect("MIMIC_ORT_DYLIB");
    let files = ModelFiles::new(models_dir());
    let config = files.config();
    // two loads of the runtime from two threads: the second must reuse the environment, not fail
    let handles: Vec<_> = gpus
        .iter()
        .map(|&g| {
            let dylib = dylib.clone();
            let files = ModelFiles::new(files.dir.clone());
            std::thread::spawn(move || -> Result<(i32, usize), String> {
                let rt = OnnxRuntime::load(&dylib, Device::Cuda(g)).map_err(|e| e.to_string())?.with_cuda_graph(true);
                let model = ModelSet::load(&rt, &files, config).map_err(|e| format!("GPU {g}: {e}"))?;
                Ok((g, model.shapes.seq_len))
            })
        })
        .collect();
    for h in handles {
        let (g, seq) = h.join().expect("thread").expect("load");
        eprintln!("GPU {g} loaded, seq_len {seq}");
    }
}
