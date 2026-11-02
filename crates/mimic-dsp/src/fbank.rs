//! `torchaudio.compliance.kaldi.fbank` with the defaults CAM++ is driven with:
//! 16 kHz, 25 ms frames / 10 ms shift, snip_edges, DC removal, pre-emphasis 0.97, povey window,
//! FFT padded to 512, power spectrum, 80 Kaldi mel bins (20 Hz .. Nyquist), natural log with `f32::EPSILON` floor.

use ndarray::Array2;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::Arc;

use crate::window::povey;

pub struct KaldiFbank {
    sample_rate: u32,
    win: usize,
    shift: usize,
    padded: usize,
    n_bins: usize,
    preemph: f32,
    window: Vec<f32>,
    /// `(n_bins, padded/2 + 1)`; the Nyquist column is zero (torchaudio pads the bank by one column).
    banks: Array2<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

fn mel_scale(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

impl KaldiFbank {
    pub fn campplus_16k() -> Self {
        Self::new(16000, 80)
    }

    pub fn new(sample_rate: u32, n_bins: usize) -> Self {
        let win = (sample_rate as f32 * 0.025) as usize; // 400
        let shift = (sample_rate as f32 * 0.010) as usize; // 160
        let padded = win.next_power_of_two(); // 512
        let n_fft_bins = padded / 2;
        let low_freq = 20.0f32;
        let high_freq = sample_rate as f32 / 2.0;
        let fft_bin_width = sample_rate as f32 / padded as f32;
        let (mel_low, mel_high) = (mel_scale(low_freq), mel_scale(high_freq));
        let delta = (mel_high - mel_low) / (n_bins as f32 + 1.0);
        let mut banks = Array2::<f32>::zeros((n_bins, n_fft_bins + 1));
        for b in 0..n_bins {
            let left = mel_low + b as f32 * delta;
            let center = mel_low + (b as f32 + 1.0) * delta;
            let right = mel_low + (b as f32 + 2.0) * delta;
            for k in 0..n_fft_bins {
                let mel = mel_scale(k as f32 * fft_bin_width);
                let up = (mel - left) / (center - left);
                let down = (right - mel) / (right - center);
                banks[(b, k)] = up.min(down).max(0.0);
            }
        }
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            sample_rate,
            win,
            shift,
            padded,
            n_bins,
            preemph: 0.97,
            window: povey(win),
            banks,
            fft: planner.plan_fft_forward(padded),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn frames_for(&self, n: usize) -> usize {
        if n < self.win {
            0
        } else {
            1 + (n - self.win) / self.shift
        }
    }

    /// `(T, n_bins)` log-mel filterbank energies.
    pub fn compute(&self, wav: &[f32]) -> Array2<f32> {
        let m = self.frames_for(wav.len());
        let n_spec = self.padded / 2 + 1;
        let mut out = Array2::<f32>::zeros((m, self.n_bins));
        let mut frame = vec![0.0f32; self.padded];
        let mut spec = vec![Complex::new(0.0f32, 0.0); n_spec];
        let mut power = vec![0.0f32; n_spec];
        let eps = f32::EPSILON;
        for t in 0..m {
            let s = &wav[t * self.shift..t * self.shift + self.win];
            // remove DC offset
            let mean = s.iter().sum::<f32>() / self.win as f32;
            for i in 0..self.win {
                frame[i] = s[i] - mean;
            }
            // pre-emphasis with replicate padding: x[0] -= c*x[0]; x[k] -= c*x[k-1]
            for i in (1..self.win).rev() {
                frame[i] -= self.preemph * frame[i - 1];
            }
            frame[0] -= self.preemph * frame[0];
            for i in 0..self.win {
                frame[i] *= self.window[i];
            }
            for i in self.win..self.padded {
                frame[i] = 0.0;
            }
            self.fft.process(&mut frame, &mut spec).expect("fft");
            for k in 0..n_spec {
                power[k] = spec[k].norm_sqr();
            }
            for b in 0..self.n_bins {
                let mut acc = 0.0f32;
                for k in 0..n_spec {
                    acc += power[k] * self.banks[(b, k)];
                }
                out[(t, b)] = acc.max(eps).ln();
            }
        }
        out
    }

    /// `compute` followed by per-bin mean subtraction over time (what seed-vc feeds CAM++).
    pub fn compute_mean_normalized(&self, wav: &[f32]) -> Array2<f32> {
        let mut f = self.compute(wav);
        let m = f.shape()[0] as f32;
        for b in 0..self.n_bins {
            let mean = f.column(b).sum() / m;
            f.column_mut(b).mapv_inplace(|v| v - mean);
        }
        f
    }
}
