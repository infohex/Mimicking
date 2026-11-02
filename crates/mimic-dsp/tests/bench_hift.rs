//! Host-side cost of the HiFT pieces at block size (241 mel frames). `cargo test -p mimic-dsp --release -- --ignored --nocapture`

use mimic_dsp::hift::{harmonic_source, istft_from_body, stft_small, SourceParams, StftParams};
use ndarray::{Array2, Array3};
use std::time::Instant;

#[test]
#[ignore]
fn hift_host_costs() {
    let t = 241usize;
    let p = SourceParams {
        sampling_rate: 22050.0,
        harmonic_num: 8,
        sine_amp: 0.1,
        noise_std: 0.003,
        voiced_threshold: 10.0,
        upsample: 256,
        linear_weight: vec![0.1; 9],
        linear_bias: 0.0,
    };
    let sp = StftParams {
        n_fft: 16,
        hop: 4,
        window: mimic_dsp::window::hann_periodic(16),
        audio_limit: 0.99,
    };
    let f0 = Array2::from_shape_fn((1, t), |(_, i)| {
        if i % 5 == 0 {
            0.0
        } else {
            180.0 + (i % 7) as f32
        }
    });
    let mut rng = rand::rng();
    let n = 20;
    let t0 = Instant::now();
    let mut src = Array2::zeros((1, t * 256));
    for _ in 0..n {
        src = harmonic_source(f0.view(), &p, &mut rng);
    }
    let a = t0.elapsed().as_secs_f32() * 1e3 / n as f32;
    let t0 = Instant::now();
    let mut st = Array3::zeros((1, 18, 1));
    for _ in 0..n {
        st = stft_small(src.view(), &sp);
    }
    let b = t0.elapsed().as_secs_f32() * 1e3 / n as f32;
    let body = Array3::from_shape_fn(st.raw_dim(), |(_, c, _)| if c < 9 { -2.0 } else { 0.3 });
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = istft_from_body(body.view(), &sp, t * 256);
    }
    let c = t0.elapsed().as_secs_f32() * 1e3 / n as f32;
    eprintln!(
        "hift host: source {a:.2} ms, stft {b:.2} ms, istft {c:.2} ms (frames {})",
        st.shape()[2]
    );
}
