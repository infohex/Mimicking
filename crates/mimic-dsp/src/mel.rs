//! Log-mel spectrogram reproducing seed-vc's `modules/audio.py::mel_spectrogram` (BigVGAN/HiFi-GAN convention):
//! reflect-pad `(n_fft - hop) / 2` on both sides, STFT with a periodic Hann window and `center=False`,
//! magnitude `sqrt(re^2 + im^2 + 1e-9)`, librosa Slaney mel filterbank, `log(max(x, 1e-5))`.

use mimic_core::MelAnalyzer;
use ndarray::Array2;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::Arc;

use crate::window::{hann_periodic, reflect_pad};

pub struct MelSpec {
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    window: Vec<f32>,
    /// `(n_mels, n_fft/2 + 1)`
    basis: Array2<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    clip: f32,
}

/// librosa Slaney mel scale (htk=False).
fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")` as `(n_mels, n_fft/2+1)`.
pub fn librosa_mel_basis(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sr as f64 / n_fft as f64)
        .collect();
    let (m_lo, m_hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut w = Array2::<f32>::zeros((n_mels, n_bins));
    for i in 0..n_mels {
        let fdiff_lo = mel_f[i + 1] - mel_f[i];
        let fdiff_hi = mel_f[i + 2] - mel_f[i + 1];
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for k in 0..n_bins {
            let lower = (fftfreqs[k] - mel_f[i]) / fdiff_lo;
            let upper = (mel_f[i + 2] - fftfreqs[k]) / fdiff_hi;
            let v = lower.min(upper).max(0.0);
            w[(i, k)] = (v * enorm) as f32;
        }
    }
    w
}

impl MelSpec {
    /// seed-vc 22.05 kHz configuration: n_fft 1024, hop 256, win 1024, 80 mels, fmin 0, fmax 8000.
    pub fn mel_22k() -> Self {
        Self::new(22050, 1024, 256, 80, 0.0, 8000.0)
    }

    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        hop: usize,
        n_mels: usize,
        fmin: f64,
        fmax: f64,
    ) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            sample_rate,
            n_fft,
            hop,
            n_mels,
            window: hann_periodic(n_fft),
            basis: librosa_mel_basis(sample_rate, n_fft, n_mels, fmin, fmax),
            fft: planner.plan_fft_forward(n_fft),
            clip: 1e-5,
        }
    }

    /// Linear magnitude spectrogram `(n_fft/2+1, T)` including the reflect padding and `+1e-9` floor.
    pub fn magnitude(&self, wav: &[f32]) -> Array2<f32> {
        let pad = (self.n_fft - self.hop) / 2;
        let x = reflect_pad(wav, pad, pad);
        let n_frames = self.frames_for(wav.len());
        let n_bins = self.n_fft / 2 + 1;
        let mut mag = Array2::<f32>::zeros((n_bins, n_frames));
        let mut buf = vec![0.0f32; self.n_fft];
        let mut spec = vec![Complex::new(0.0f32, 0.0); n_bins];
        for t in 0..n_frames {
            let s = t * self.hop;
            for i in 0..self.n_fft {
                buf[i] = x[s + i] * self.window[i];
            }
            self.fft.process(&mut buf, &mut spec).expect("fft");
            for k in 0..n_bins {
                mag[(k, t)] = (spec[k].norm_sqr() + 1e-9).sqrt();
            }
        }
        mag
    }
}

impl MelAnalyzer for MelSpec {
    fn analyze(&self, wav: &[f32]) -> Array2<f32> {
        let mag = self.magnitude(wav);
        let mut mel = self.basis.dot(&mag);
        mel.mapv_inplace(|v| v.max(self.clip).ln());
        mel
    }
    fn n_mels(&self) -> usize {
        self.n_mels
    }
    fn hop(&self) -> usize {
        self.hop
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn frames_for(&self, n_samples: usize) -> usize {
        // padded length = n + (n_fft - hop); frames = (padded - n_fft) / hop + 1 = (n - hop) / hop + 1
        (n_samples + (self.n_fft - self.hop) - self.n_fft) / self.hop + 1
    }
}
