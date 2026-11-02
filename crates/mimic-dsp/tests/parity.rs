//! Parity against the PyTorch fixtures that `mimictrain.export` writes into `<set>/fixtures`
//! (`MIMIC_MODELS=<set>` or `MIMIC_FIXTURES=<dir>`); tests are skipped (not failed) when fixtures are absent.

use mimic_core::MelAnalyzer;
use mimic_dsp::hift::{harmonic_source, istft_from_body, stft_small, SourceParams, StftParams};
use mimic_dsp::{KaldiFbank, MelSpec, SincResampler};
use ndarray::{Array2, ArrayD, Ix1, Ix2, Ix3, IxDyn};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
/// Fixtures: `MIMIC_FIXTURES`, else `<models>/fixtures` (written by mimictrain.export next to the graphs).
fn fixture_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MIMIC_FIXTURES") {
        return PathBuf::from(d);
    }
    models_dir().join("fixtures")
}

fn models_dir() -> PathBuf {
    std::env::var_os("MIMIC_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("models/Mimicking-v2.0-lowlat"))
}

fn open(name: &str) -> Option<NpzReader<File>> {
    let p = fixture_dir().join(name);
    if !p.exists() {
        eprintln!("fixture {} missing, skipping", p.display());
        return None;
    }
    Some(NpzReader::new(File::open(p).unwrap()).unwrap())
}

fn f32_arr(r: &mut NpzReader<File>, name: &str) -> ArrayD<f32> {
    r.by_name::<ndarray::OwnedRepr<f32>, IxDyn>(&format!("{name}.npy"))
        .or_else(|_| r.by_name::<ndarray::OwnedRepr<f32>, IxDyn>(name))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// numpy scalars: int64 on Linux, int32 on Windows (numpy 1.x), float64 for floats
fn scalar(r: &mut NpzReader<File>, name: &str) -> f64 {
    let n = format!("{name}.npy");
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<i64>, IxDyn>(&n) {
        return a.iter().next().copied().unwrap() as f64;
    }
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<i32>, IxDyn>(&n) {
        return a.iter().next().copied().unwrap() as f64;
    }
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<f64>, IxDyn>(&n) {
        return a.iter().next().copied().unwrap();
    }
    let a = r
        .by_name::<ndarray::OwnedRepr<f32>, IxDyn>(&n)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    a.iter().next().copied().unwrap() as f64
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    num / (den + 1e-12)
}

#[test]
fn resample_matches_torchaudio() {
    let Some(mut r) = open("campplus.npz") else {
        return;
    };
    let ref22 = f32_arr(&mut r, "ref22")
        .into_dimensionality::<Ix1>()
        .unwrap();
    let ref16 = f32_arr(&mut r, "ref16")
        .into_dimensionality::<Ix1>()
        .unwrap();
    let rs = SincResampler::new(22050, 16000);
    let y = rs.resample(ref22.as_slice().unwrap());
    assert_eq!(y.len(), ref16.len());
    let d = max_abs_diff(&y, ref16.as_slice().unwrap());
    eprintln!("resample max abs diff {d:e}");
    assert!(d < 1e-4, "resample differs from torchaudio: {d}");
}

#[test]
fn mel_matches_reference() {
    let Some(mut r) = open("campplus.npz") else {
        return;
    };
    let ref22 = f32_arr(&mut r, "ref22")
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mel2 = f32_arr(&mut r, "mel2")
        .into_dimensionality::<Ix3>()
        .unwrap(); // (1, 80, 258)
    let m = MelSpec::mel_22k();
    assert_eq!(m.frames_for(ref22.len()), mel2.shape()[2]);
    let mine = m.analyze(ref22.as_slice().unwrap());
    let theirs: Array2<f32> = mel2.index_axis(ndarray::Axis(0), 0).to_owned();
    let d = max_abs_diff(
        mine.as_slice().unwrap(),
        theirs.as_standard_layout().as_slice().unwrap(),
    );
    let rl = rel_l2(
        mine.as_slice().unwrap(),
        theirs.as_standard_layout().as_slice().unwrap(),
    );
    eprintln!("mel max abs diff {d:e} rel l2 {rl:e}");
    assert!(d < 2e-3, "mel differs: max {d}");
}

#[test]
fn fbank_matches_torchaudio_kaldi() {
    let Some(mut r) = open("campplus.npz") else {
        return;
    };
    let ref16 = f32_arr(&mut r, "ref16")
        .into_dimensionality::<Ix1>()
        .unwrap();
    let fb = f32_arr(&mut r, "fbank")
        .into_dimensionality::<Ix3>()
        .unwrap(); // (1, T, 80), mean-normalised
    let k = KaldiFbank::campplus_16k();
    let mine = k.compute_mean_normalized(ref16.as_slice().unwrap());
    assert_eq!(mine.shape(), &fb.shape()[1..]);
    let theirs: Array2<f32> = fb.index_axis(ndarray::Axis(0), 0).to_owned();
    let d = max_abs_diff(
        mine.as_slice().unwrap(),
        theirs.as_standard_layout().as_slice().unwrap(),
    );
    let rl = rel_l2(
        mine.as_slice().unwrap(),
        theirs.as_standard_layout().as_slice().unwrap(),
    );
    eprintln!("fbank max abs diff {d:e} rel l2 {rl:e}");
    assert!(d < 2e-3, "fbank differs: max {d}");
}

fn stft_params() -> Option<StftParams> {
    let p = models_dir().join("hift_source.npz");
    if !p.exists() {
        eprintln!("{} missing, skipping", p.display());
        return None;
    }
    let mut r = NpzReader::new(File::open(p).unwrap()).unwrap();
    let window = f32_arr(&mut r, "window")
        .into_dimensionality::<Ix1>()
        .unwrap()
        .to_vec();
    let n_fft = scalar(&mut r, "n_fft") as usize;
    let hop = scalar(&mut r, "hop_len") as usize;
    let audio_limit = scalar(&mut r, "audio_limit") as f32;
    Some(StftParams {
        n_fft,
        hop,
        window,
        audio_limit,
    })
}

#[test]
fn hift_stft_matches_torch() {
    let (Some(mut r), Some(p)) = (open("hift.npz"), stft_params()) else {
        return;
    };
    let source = f32_arr(&mut r, "source"); // (1, 1, L)
    let s_stft = f32_arr(&mut r, "s_stft")
        .into_dimensionality::<Ix3>()
        .unwrap(); // (1, 18, frames)
    let l = *source.shape().last().unwrap();
    let src2 = source.into_shape_with_order((1, l)).unwrap();
    let mine = stft_small(src2.view(), &p);
    assert_eq!(mine.shape(), s_stft.shape());
    let d = max_abs_diff(
        mine.as_slice().unwrap(),
        s_stft.as_standard_layout().as_slice().unwrap(),
    );
    eprintln!("stft16 max abs diff {d:e}");
    assert!(d < 1e-5, "stft differs: {d}");
}

#[test]
fn hift_istft_matches_torch() {
    let (Some(mut r), Some(p)) = (open("hift.npz"), stft_params()) else {
        return;
    };
    let body = f32_arr(&mut r, "body_x")
        .into_dimensionality::<Ix3>()
        .unwrap();
    let wav = f32_arr(&mut r, "wav").into_dimensionality::<Ix2>().unwrap(); // (1, L)
    let mine = istft_from_body(body.view(), &p, wav.shape()[1]);
    let d = max_abs_diff(
        mine.as_slice().unwrap(),
        wav.as_standard_layout().as_slice().unwrap(),
    );
    let rl = rel_l2(
        mine.as_slice().unwrap(),
        wav.as_standard_layout().as_slice().unwrap(),
    );
    eprintln!("istft16 max abs diff {d:e} rel l2 {rl:e}");
    assert!(d < 2e-3, "istft differs: {d}");
}

#[test]
fn harmonic_source_shape_and_energy() {
    let (Some(mut r), Some(_p)) = (open("hift.npz"), stft_params()) else {
        return;
    };
    let f0 = f32_arr(&mut r, "f0").into_dimensionality::<Ix2>().unwrap(); // (1, T)
    let source = f32_arr(&mut r, "source");
    let sp = models_dir().join("hift_source.npz");
    let mut s = NpzReader::new(File::open(sp).unwrap()).unwrap();
    let w = f32_arr(&mut s, "l_linear_weight")
        .into_dimensionality::<Ix2>()
        .unwrap();
    let b = f32_arr(&mut s, "l_linear_bias")
        .into_dimensionality::<Ix1>()
        .unwrap();
    let params = SourceParams {
        sampling_rate: scalar(&mut s, "sampling_rate") as f32,
        harmonic_num: scalar(&mut s, "harmonic_num") as usize,
        sine_amp: scalar(&mut s, "sine_amp") as f32,
        noise_std: scalar(&mut s, "noise_std") as f32,
        voiced_threshold: scalar(&mut s, "voiced_threshold") as f32,
        upsample: scalar(&mut s, "upsample") as usize,
        linear_weight: w.row(0).to_vec(),
        linear_bias: b[0],
    };
    let mut rng = rand::rng();
    let mine = harmonic_source(f0.view(), &params, &mut rng);
    assert_eq!(mine.shape()[1], *source.shape().last().unwrap());
    // random phases differ, so compare RMS over the utterance only
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let (a, b) = (
        rms(mine.as_slice().unwrap()),
        rms(source.as_slice().unwrap()),
    );
    eprintln!("source rms mine {a:.4} torch {b:.4}");
    assert!(
        (a - b).abs() / b < 0.15,
        "source energy differs: {a} vs {b}"
    );
}
