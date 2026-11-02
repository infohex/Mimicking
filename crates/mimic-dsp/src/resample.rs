//! Windowed-sinc resampler reproducing `torchaudio.functional.resample(..., resampling_method="sinc_interp_hann")`
//! with `lowpass_filter_width=6`, `rolloff=0.99`.
//!
//! The kernel bank has one FIR per output phase (`new` phases after reducing by the gcd); the input is
//! zero-padded by `width` on the left and `width + orig` on the right and the filters are applied with
//! stride `orig`, exactly as torchaudio's `conv1d` formulation does.

use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct SincResampler {
    orig: usize,
    new: usize,
    width: usize,
    /// `new` kernels of length `2 * width + orig`.
    kernels: Vec<Vec<f32>>,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl SincResampler {
    pub fn new(orig_freq: usize, new_freq: usize) -> Self {
        Self::with_params(orig_freq, new_freq, 6, 0.99)
    }

    pub fn with_params(
        orig_freq: usize,
        new_freq: usize,
        lowpass_filter_width: usize,
        rolloff: f32,
    ) -> Self {
        let g = gcd(orig_freq, new_freq);
        let orig = orig_freq / g;
        let new = new_freq / g;
        let base_freq = (orig.min(new) as f32) * rolloff;
        let width = ((lowpass_filter_width as f32) * orig as f32 / base_freq).ceil() as usize;
        let klen = 2 * width + orig;
        let lw = lowpass_filter_width as f32;
        let scale = base_freq / orig as f32;
        let mut kernels = Vec::with_capacity(new);
        for i in 0..new {
            let mut k = Vec::with_capacity(klen);
            for j in 0..klen {
                // idx = (j - width) / orig ; t = -i / new + idx
                let idx = (j as f32 - width as f32) / orig as f32;
                let mut t = -(i as f32) / new as f32 + idx;
                t *= base_freq;
                t = t.clamp(-lw, lw);
                let window = (t * PI / lw / 2.0).cos().powi(2);
                let t = t * PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                k.push(sinc * window * scale);
            }
            kernels.push(k);
        }
        Self {
            orig,
            new,
            width,
            kernels,
        }
    }

    /// Output length for an input of `n` samples (`ceil(new * n / orig)`).
    pub fn output_len(&self, n: usize) -> usize {
        (self.new * n).div_ceil(self.orig)
    }

    pub fn resample(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let klen = 2 * self.width + self.orig;
        // padded = [0]*width ++ x ++ [0]*(width + orig)
        let padded_len = n + 2 * self.width + self.orig;
        let get = |p: usize| -> f32 {
            if p < self.width || p >= self.width + n {
                0.0
            } else {
                x[p - self.width]
            }
        };
        let n_frames = (padded_len - klen) / self.orig + 1;
        let target = self.output_len(n);
        let mut out = vec![0.0f32; n_frames * self.new];
        for m in 0..n_frames {
            let base = m * self.orig;
            for (i, k) in self.kernels.iter().enumerate() {
                let mut acc = 0.0f32;
                for j in 0..klen {
                    acc += k[j] * get(base + j);
                }
                out[m * self.new + i] = acc;
            }
        }
        out.truncate(target);
        out
    }
}

impl mimic_core::Resampler for SincResampler {
    fn resample(&self, x: &[f32]) -> Vec<f32> {
        SincResampler::resample(self, x)
    }
    fn output_len(&self, n_samples: usize) -> usize {
        SincResampler::output_len(self, n_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passthrough_length() {
        let r = SincResampler::new(22050, 16000);
        assert_eq!(r.output_len(22050), 16000);
        let y = r.resample(&vec![0.0; 22050]);
        assert_eq!(y.len(), 16000);
    }
}
