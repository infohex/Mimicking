//! Stage-by-stage parity of the ONNX-backed stages against the PyTorch fixtures, plus the 10-step solver
//! and the full HiFT chain. Needs a GPU and an onnxruntime dylib:
//!   MIMIC_ORT_DYLIB=<path to onnxruntime.dll>  MIMIC_GPU=1  cargo test -p mimic-model --release -- --nocapture
//! Tests are skipped (not failed) when the env var, fixtures or graphs are missing.

use mimic_core::{ContentEncoder, LengthRegulator, Prompt, SpeakerEncoder, VelocityEstimator};
use mimic_onnx::{Device, OnnxRuntime};
use mimic_pipeline::EulerCfg;
use mimic_model::{ModelFiles, ModelSet};
use ndarray::{s, Array3, ArrayD, Ix1, Ix2, Ix3, IxDyn};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `MIMIC_MODELS` (a model directory) or `models/Mimicking-v2.0-lowlat`.
fn models_dir() -> PathBuf {
    std::env::var_os("MIMIC_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("models/Mimicking-v2.0-lowlat"))
}
/// Fixtures: `MIMIC_FIXTURES`, else `<models>/fixtures` (written by mimictrain.export next to the graphs).
fn fixture_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MIMIC_FIXTURES") {
        return PathBuf::from(d);
    }
    models_dir().join("fixtures")
}
fn open(name: &str) -> Option<NpzReader<File>> {
    let p = fixture_dir().join(name);
    if !p.exists() {
        eprintln!("fixture {} missing, skipping", p.display());
        return None;
    }
    Some(NpzReader::new(File::open(p).unwrap()).unwrap())
}

fn arr(r: &mut NpzReader<File>, name: &str) -> ArrayD<f32> {
    r.by_name::<ndarray::OwnedRepr<f32>, IxDyn>(&format!("{name}.npy"))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    (max, mean, num / (den + 1e-12))
}

fn flat(a: &ArrayD<f32>) -> Vec<f32> {
    a.as_standard_layout().iter().copied().collect()
}

static MODELS: OnceLock<Option<Mutex<ModelSet>>> = OnceLock::new();

fn models() -> Option<&'static Mutex<ModelSet>> {
    MODELS
        .get_or_init(|| {
            let Some(dylib) = std::env::var_os("MIMIC_ORT_DYLIB") else {
                eprintln!("MIMIC_ORT_DYLIB not set, skipping ORT tests");
                return None;
            };
            let gpu: i32 = std::env::var("MIMIC_GPU")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let files = ModelFiles::new(models_dir());
            if !files.dit_step().exists() {
                eprintln!("graphs missing in {}, skipping", files.dir.display());
                return None;
            }
            let cuda_graph = std::env::var("MIMIC_CUDA_GRAPH")
                .map(|v| v != "0")
                .unwrap_or(true);
            eprintln!("cuda graph: {cuda_graph}");
            let rt = OnnxRuntime::load(dylib, Device::Cuda(gpu))
                .expect("load onnxruntime")
                .with_cuda_graph(cuda_graph);
            let t0 = std::time::Instant::now();
            let config = files.config(); // the directory's manifest carries its geometry
            let m = ModelSet::load(&rt, &files, config).expect("load models");
            eprintln!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());
            Some(Mutex::new(m))
        })
        .as_ref()
}

#[test]
fn xlsr_graph_matches_fixture() {
    let (Some(m), Some(mut r)) = (models(), open("xlsr.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let x = arr(&mut r, "input_values")
        .into_dimensionality::<Ix2>()
        .unwrap();
    let y_ref = arr(&mut r, "hidden");
    let y = m.encoder.encode(x.view()).unwrap();
    assert_eq!(m.encoder.frames_for(x.shape()[1]), y.shape()[1]);
    let (mx, mean, rl) = stats(&flat(&y.into_dyn()), &flat(&y_ref));
    eprintln!("xlsr max {mx:e} mean {mean:e} rel_l2 {rl:e}");
    assert!(rl < 1e-2, "xlsr rel l2 {rl}");
}

#[test]
fn length_regulator_graph_matches_fixture() {
    let (Some(m), Some(mut r)) = (models(), open("length_regulator.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let c = arr(&mut r, "content").into_dimensionality::<Ix3>().unwrap();
    let y_ref = arr(&mut r, "cond");
    let y = m.regulator.regulate(c.view(), y_ref.shape()[1]).unwrap();
    let (mx, mean, rl) = stats(&flat(&y.into_dyn()), &flat(&y_ref));
    eprintln!("length_regulator max {mx:e} mean {mean:e} rel_l2 {rl:e}");
    // fp32 graph, but the CUDA EP runs fp32 MatMul in TF32 by default; with the 2.0 content codebook (a softmax over
    // 4096 squared distances) that rounding shows as ~2-3e-3 (6.5e-4 in exact fp32). The baseline regulator sits at 4e-4.
    assert!(rl < 5e-3, "regulator rel l2 {rl}");
}

#[test]
fn campplus_from_waveform_matches_fixture() {
    let (Some(m), Some(mut r)) = (models(), open("campplus.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let ref16 = arr(&mut r, "ref16").into_dimensionality::<Ix1>().unwrap();
    let style_ref = arr(&mut r, "style");
    let style = m.speaker.embed(ref16.as_slice().unwrap()).unwrap();
    let (mx, mean, rl) = stats(style.as_slice().unwrap(), &flat(&style_ref));
    // cosine similarity is what matters for a speaker embedding
    let a = style.as_slice().unwrap();
    let b = flat(&style_ref);
    let cos = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>()
        / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
            * b.iter().map(|y| y * y).sum::<f32>().sqrt());
    eprintln!("campplus max {mx:e} mean {mean:e} rel_l2 {rl:e} cosine {cos:.6}");
    assert!(cos > 0.999, "campplus cosine {cos}");
}

#[test]
fn dit_step_graph_matches_fixture() {
    let (Some(m), Some(mut r)) = (models(), open("dit_step.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let x = arr(&mut r, "x").into_dimensionality::<Ix3>().unwrap();
    let px = arr(&mut r, "prompt_x")
        .into_dimensionality::<Ix3>()
        .unwrap();
    let t = arr(&mut r, "t").into_dimensionality::<Ix1>().unwrap();
    let st = arr(&mut r, "style").into_dimensionality::<Ix2>().unwrap();
    let mu = arr(&mut r, "mu").into_dimensionality::<Ix3>().unwrap();
    let v_ref = arr(&mut r, "v");
    let v = m
        .estimator
        .velocity(x.view(), px.view(), t.view(), st.view(), mu.view())
        .unwrap();
    let (mx, mean, rl) = stats(&flat(&v.into_dyn()), &flat(&v_ref));
    eprintln!("dit_step max {mx:e} mean {mean:e} rel_l2 {rl:e}");
    assert!(rl < 5e-3, "dit_step rel l2 {rl}");
}

#[test]
fn euler_cfg_10_steps_matches_torch_cfm() {
    let (Some(m), Some(mut r)) = (models(), open("cfm.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let z = arr(&mut r, "z").into_dimensionality::<Ix3>().unwrap();
    let mu = arr(&mut r, "cat_condition")
        .into_dimensionality::<Ix3>()
        .unwrap();
    let style = arr(&mut r, "style").into_dimensionality::<Ix2>().unwrap();
    let mel2 = arr(&mut r, "mel2").into_dimensionality::<Ix3>().unwrap();
    let mel_ref = arr(&mut r, "mel_out");
    let p = mel2.shape()[2];
    let prompt = Prompt {
        cond: Array3::zeros((1, p, 1)),
        mel: mel2.clone(),
        style,
    };
    let solver = EulerCfg {
        steps: 10,
        cfg: 0.7,
    };
    let t0 = std::time::Instant::now();
    let x = solver
        .solve(&mut m.estimator, mu.view(), &prompt, &z)
        .unwrap();
    let ms = t0.elapsed().as_secs_f32() * 1e3;
    let mel_out = x.slice(s![.., .., p..]).to_owned();
    let (mx, mean, rl) = stats(&flat(&mel_out.into_dyn()), &flat(&mel_ref));
    eprintln!("cfm 10 steps: {ms:.1} ms; max {mx:e} mean {mean:e} rel_l2 {rl:e}");
    assert!(rl < 5e-3, "cfm rel l2 {rl}");
}

#[test]
fn fused_euler_10_steps_matches_torch_cfm() {
    let (Some(m), Some(mut r)) = (models(), open("cfm.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let Some(stepper) = m.stepper.as_mut() else {
        eprintln!("no dit_step_fused.onnx, skipping");
        return;
    };
    let z = arr(&mut r, "z").into_dimensionality::<Ix3>().unwrap();
    let mu = arr(&mut r, "cat_condition")
        .into_dimensionality::<Ix3>()
        .unwrap();
    let style = arr(&mut r, "style").into_dimensionality::<Ix2>().unwrap();
    let mel2 = arr(&mut r, "mel2").into_dimensionality::<Ix3>().unwrap();
    let mel_ref = arr(&mut r, "mel_out");
    let p = mel2.shape()[2];
    let prompt = Prompt {
        cond: Array3::zeros((1, p, 1)),
        mel: mel2.clone(),
        style,
    };
    let solver = EulerCfg {
        steps: 10,
        cfg: 0.7,
    };
    let t0 = std::time::Instant::now();
    let x = solver.solve_fused(stepper, mu.view(), &prompt, &z).unwrap();
    let ms = t0.elapsed().as_secs_f32() * 1e3;
    let mel_out = x.slice(s![.., .., p..]).to_owned();
    let (mx, mean, rl) = stats(&flat(&mel_out.into_dyn()), &flat(&mel_ref));
    eprintln!("fused cfm 10 steps: {ms:.1} ms; max {mx:e} mean {mean:e} rel_l2 {rl:e}");
    assert!(rl < 5e-3, "fused cfm rel l2 {rl}");
}

#[test]
fn hift_chain_matches_torch_with_shared_source() {
    let (Some(m), Some(mut r)) = (models(), open("hift.npz")) else {
        return;
    };
    let mut m = m.lock().unwrap_or_else(|p| p.into_inner());
    let mel = arr(&mut r, "mel").into_dimensionality::<Ix3>().unwrap();
    let f0_ref = arr(&mut r, "f0");
    let source = arr(&mut r, "source");
    let wav_ref = arr(&mut r, "wav");
    let f0 = m.vocoder.predict_f0(mel.view()).unwrap();
    let (mx, _, rl) = stats(&flat(&f0.into_dyn()), &flat(&f0_ref));
    eprintln!("hift f0 max {mx:e} rel_l2 {rl:e}");
    assert!(rl < 1e-2, "f0 rel l2 {rl}");
    let l = *source.shape().last().unwrap();
    let src2 = source.into_shape_with_order((1, l)).unwrap();
    let t0 = std::time::Instant::now();
    let wav = m
        .vocoder
        .vocode_with_source(mel.view(), src2.view())
        .unwrap();
    let ms = t0.elapsed().as_secs_f32() * 1e3;
    let (mx, mean, rl) = stats(&flat(&wav.into_dyn()), &flat(&wav_ref));
    eprintln!(
        "hift wav ({ms:.1} ms incl. host stft/istft): max {mx:e} mean {mean:e} rel_l2 {rl:e}"
    );
    assert!(rl < 1e-2, "hift wav rel l2 {rl}");
}
