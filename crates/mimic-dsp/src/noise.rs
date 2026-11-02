//! Gaussian samples without pulling in `rand_distr`.

use rand::Rng;

/// Two independent standard-normal samples (Box–Muller).
#[inline]
pub fn standard_normal_pair<R: Rng + ?Sized>(rng: &mut R) -> (f32, f32) {
    let u1: f32 = rng.random_range(f32::MIN_POSITIVE..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (2.0 * std::f32::consts::PI * u2).sin_cos();
    (r * c, r * s)
}

/// One standard-normal sample (the second value of the Box–Muller pair is discarded).
pub fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    standard_normal_pair(rng).0
}
