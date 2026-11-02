//! Analysis windows.

use std::f32::consts::PI;

/// `torch.hann_window(n)` (periodic): `0.5 - 0.5 cos(2 pi k / n)`.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * PI * k as f32 / n as f32).cos())
        .collect()
}

/// `torch.hann_window(n, periodic=False)` / scipy `hann(n, sym=True)`: `0.5 - 0.5 cos(2 pi k / (n - 1))`.
pub fn hann_symmetric(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * PI * k as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

/// Kaldi "povey" window: symmetric Hann raised to 0.85.
pub fn povey(n: usize) -> Vec<f32> {
    hann_symmetric(n)
        .into_iter()
        .map(|w| w.powf(0.85))
        .collect()
}

/// Reflect-pad (`numpy.pad(mode="reflect")` / `torch.nn.functional.pad(mode="reflect")`): no edge repeat.
pub fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    assert!(n > left && n > right, "reflect pad larger than signal");
    let mut out = Vec::with_capacity(n + left + right);
    for i in (1..=left).rev() {
        out.push(x[i]);
    }
    out.extend_from_slice(x);
    for i in 1..=right {
        out.push(x[n - 1 - i]);
    }
    out
}
