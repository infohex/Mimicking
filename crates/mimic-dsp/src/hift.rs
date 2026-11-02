//! The parts of the HiFT vocoder that stay outside the neural graph:
//! harmonic-plus-noise source (`SourceModuleHnNSF`), the 16-point STFT of that source, and the final ISTFT.
//!
//! With `n_fft = 16` the DFT is done as a dense matrix product, which is cheaper than an FFT at this size
//! and matches the PyTorch reference fixtures (`hift.npz`, written by `mimictrain.export.shared`) (parity 1e-7 for STFT, 5e-4 for ISTFT).
//! Hot loops run on plain slices; one block (241 mel frames, 61 696 samples) costs a few milliseconds.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rand::Rng;
use std::f32::consts::PI;

/// Constants of HiFT's source module (saved by `mimictrain.export.shared` into `hift_source.npz`).
#[derive(Debug, Clone)]
pub struct SourceParams {
    pub sampling_rate: f32,
    /// number of overtones; `harmonic_num + 1` sine channels
    pub harmonic_num: usize,
    pub sine_amp: f32,
    pub noise_std: f32,
    pub voiced_threshold: f32,
    /// mel frame -> samples upsampling factor (256 for 22.05 kHz HiFT)
    pub upsample: usize,
    /// `l_linear` weight `(harmonic_num + 1,)` and bias
    pub linear_weight: Vec<f32>,
    pub linear_bias: f32,
}

#[derive(Debug, Clone)]
pub struct StftParams {
    pub n_fft: usize,
    pub hop: usize,
    pub window: Vec<f32>,
    pub audio_limit: f32,
}

/// Harmonic source for a batch of frame-rate f0 tracks `(B, T)` -> `(B, T * upsample)`.
///
/// Random overtone phases and the additive noise come from `rng`; the fundamental keeps phase 0 like upstream.
/// The per-harmonic noise terms are independent Gaussians with a common amplitude, so their weighted sum is
/// drawn directly as one Gaussian with variance `sum(w_k^2)`: identical distribution, 9x fewer samples.
pub fn harmonic_source<R: Rng + ?Sized>(
    f0: ArrayView2<'_, f32>,
    p: &SourceParams,
    rng: &mut R,
) -> Array2<f32> {
    let (b, t) = f0.dim();
    let up = p.upsample;
    let l = t * up;
    let h = p.harmonic_num + 1;
    let w = &p.linear_weight;
    let w_norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut out = Array2::<f32>::zeros((b, l));
    let mut phase = vec![0.0f32; h];
    let mut cum = vec![0.0f64; h];
    let mut sines = vec![0.0f32; l]; // weighted sum of harmonic sines
    for bi in 0..b {
        for k in 1..h {
            phase[k] = rng.random_range(-PI..PI);
        }
        phase[0] = 0.0;
        cum.iter_mut().for_each(|c| *c = 0.0);
        sines.iter_mut().for_each(|s| *s = 0.0);
        let f0_row = f0.row(bi);
        // harmonic-major loop: each harmonic is one sequential cumsum over the samples
        for k in 0..h {
            let step_scale = (k as f32 + 1.0) / p.sampling_rate;
            let wk = p.sine_amp * w[k];
            let ph = phase[k];
            let mut c = 0.0f64;
            for (fi, &f) in f0_row.iter().enumerate() {
                let inc = (f * step_scale) as f64;
                let voiced = f > p.voiced_threshold;
                let seg = &mut sines[fi * up..(fi + 1) * up];
                if voiced {
                    // within a frame the increment is constant: rotate a unit phasor instead of calling sin()
                    // per sample; re-anchor on the exact cumulative phase at every frame start
                    let delta = 2.0 * PI * inc as f32;
                    let (ds, dc) = delta.sin_cos();
                    c += inc;
                    let theta0 = 2.0 * PI * (c.rem_euclid(1.0) as f32) + ph;
                    let (mut sn, mut cs) = theta0.sin_cos();
                    seg[0] += wk * sn;
                    for s in seg.iter_mut().skip(1) {
                        let ns = sn * dc + cs * ds;
                        let nc = cs * dc - sn * ds;
                        sn = ns;
                        cs = nc;
                        *s += wk * sn;
                    }
                    c += inc * (up as f64 - 1.0);
                } else {
                    // unvoiced: sine contribution is zeroed by uv, but the phase still advances
                    c += inc * up as f64;
                }
            }
            cum[k] = c;
        }
        let mut row = out.row_mut(bi);
        let row = row.as_slice_mut().expect("contiguous");
        let mut pending: Option<f32> = None;
        for (fi, &f) in f0_row.iter().enumerate() {
            let uv = f > p.voiced_threshold;
            let noise_amp = if uv { p.noise_std } else { p.sine_amp / 3.0 } * w_norm;
            for n in fi * up..(fi + 1) * up {
                let g = match pending.take() {
                    Some(v) => v,
                    None => {
                        let (a, bb) = crate::noise::standard_normal_pair(rng);
                        pending = Some(bb);
                        a
                    }
                };
                row[n] = (sines[n] + noise_amp * g + p.linear_bias).tanh();
            }
        }
    }
    out
}

/// Real DFT matrices, row-major `n_fft x bins`: cos and sin of `-2 pi n k / N`.
fn dft_mats(n_fft: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let bins = n_fft / 2 + 1;
    let mut c = vec![0.0f32; n_fft * bins];
    let mut s = vec![0.0f32; n_fft * bins];
    for n in 0..n_fft {
        for k in 0..bins {
            let ang = -2.0 * PI * (n * k) as f32 / n_fft as f32;
            c[n * bins + k] = ang.cos();
            s[n * bins + k] = ang.sin();
        }
    }
    (c, s, bins)
}

/// `torch.stft(x, n_fft, hop, win_length=n_fft, window, center=True, pad_mode="reflect", onesided=True)`
/// for `(B, L)` -> `(B, 2 * (n_fft/2+1), frames)` with real parts first, imaginary parts second.
pub fn stft_small(x: ArrayView2<'_, f32>, p: &StftParams) -> Array3<f32> {
    let (b, l) = x.dim();
    let n_fft = p.n_fft;
    let pad = n_fft / 2;
    let n_frames = 1 + (l + 2 * pad - n_fft) / p.hop;
    let (c, s, bins) = dft_mats(n_fft);
    let mut out = Array3::<f32>::zeros((b, 2 * bins, n_frames));
    let mut frame = vec![0.0f32; n_fft];
    for bi in 0..b {
        let row = x.row(bi);
        let padded = crate::window::reflect_pad(row.as_slice().expect("contiguous"), pad, pad);
        let mut o = out.index_axis_mut(ndarray::Axis(0), bi);
        let o = o.as_slice_mut().expect("contiguous"); // (2*bins, frames) row-major
        for f in 0..n_frames {
            let st = f * p.hop;
            for n in 0..n_fft {
                frame[n] = padded[st + n] * p.window[n];
            }
            for k in 0..bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for n in 0..n_fft {
                    let v = frame[n];
                    re += v * c[n * bins + k];
                    im += v * s[n * bins + k];
                }
                o[k * n_frames + f] = re;
                o[(bins + k) * n_frames + f] = im;
            }
        }
    }
    out
}

/// HiFT's final stage: `x = conv_post output (B, n_fft+2, frames)` -> magnitude `exp(x[:bins])` clipped at 1e2,
/// phase `sin(x[bins:])`, `torch.istft(center=True)` to `length` samples, then clamp to `+-audio_limit`.
pub fn istft_from_body(body: ArrayView3<'_, f32>, p: &StftParams, length: usize) -> Array2<f32> {
    let (b, ch, n_frames) = body.dim();
    let n_fft = p.n_fft;
    let (c, s, bins) = dft_mats(n_fft);
    assert_eq!(ch, 2 * bins, "body output must have n_fft + 2 channels");
    let pad = n_fft / 2;
    let total = n_fft + p.hop * (n_frames - 1);
    let mut wsum = vec![0.0f32; total];
    for f in 0..n_frames {
        for n in 0..n_fft {
            wsum[f * p.hop + n] += p.window[n] * p.window[n];
        }
    }
    let mut wk = vec![2.0f32; bins];
    wk[0] = 1.0;
    wk[bins - 1] = 1.0;
    let inv_n = 1.0 / n_fft as f32;
    let mut out = Array2::<f32>::zeros((b, length));
    let mut y = vec![0.0f32; total];
    let mut re = vec![0.0f32; bins];
    let mut im = vec![0.0f32; bins];
    let body_std = body.as_standard_layout();
    let body_s = body_std.as_slice().expect("contiguous");
    for bi in 0..b {
        y.iter_mut().for_each(|v| *v = 0.0);
        let base = bi * ch * n_frames;
        for f in 0..n_frames {
            for k in 0..bins {
                let mag = body_s[base + k * n_frames + f].exp().min(1e2);
                let ph = body_s[base + (bins + k) * n_frames + f].sin();
                let (sn, cs) = ph.sin_cos();
                re[k] = mag * cs * wk[k];
                im[k] = mag * sn * wk[k];
            }
            let yf = &mut y[f * p.hop..f * p.hop + n_fft];
            for n in 0..n_fft {
                // x[n] = 1/N sum_k w_k (Re cos(2pi nk/N) - Im sin(2pi nk/N)); S holds sin(-2pi nk/N)
                let cn = &c[n * bins..(n + 1) * bins];
                let sn_ = &s[n * bins..(n + 1) * bins];
                let mut acc = 0.0f32;
                for k in 0..bins {
                    acc += re[k] * cn[k] + im[k] * sn_[k];
                }
                yf[n] += acc * inv_n * p.window[n];
            }
        }
        let mut row = out.row_mut(bi);
        let row = row.as_slice_mut().expect("contiguous");
        for i in 0..length {
            let v = y[pad + i] / wsum[pad + i].max(1e-11);
            row[i] = v.clamp(-p.audio_limit, p.audio_limit);
        }
    }
    out
}
