//! SOLA (synchronous overlap-add) block stitching, as used by the real-time GUIs of RVC / DDSP-SVC / seed-vc:
//! the newly synthesised block is shifted by the lag that best correlates with the tail of the previous block,
//! then cross-faded with a sin^2 window.

#[derive(Debug, Clone)]
pub struct Sola {
    block: usize,
    buf_len: usize,
    search: usize,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,
    /// tail of the previous output (`buf_len` samples), already faded appropriately
    buffer: Vec<f32>,
    primed: bool,
}

impl Sola {
    /// `block`: output samples per hop; `buf_len`: cross-fade length; `search`: lag search range.
    pub fn new(block: usize, buf_len: usize, search: usize) -> Self {
        let fade_in: Vec<f32> = (0..buf_len)
            .map(|i| {
                let x = if buf_len > 1 {
                    i as f32 / (buf_len as f32 - 1.0)
                } else {
                    1.0
                };
                (0.5 * std::f32::consts::PI * x).sin().powi(2)
            })
            .collect();
        let fade_out = fade_in.iter().map(|v| 1.0 - v).collect();
        Self {
            block,
            buf_len,
            search,
            fade_in,
            fade_out,
            buffer: vec![0.0; buf_len],
            primed: false,
        }
    }

    /// Minimum length `infer` must have: `block + buf_len + search`.
    pub fn needed_input(&self) -> usize {
        self.block + self.buf_len + self.search
    }

    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        self.primed = false;
    }

    /// Consume a freshly synthesised segment (at least [`needed_input`](Self::needed_input) samples) and return
    /// exactly `block` output samples.
    pub fn process(&mut self, infer: &[f32]) -> Vec<f32> {
        assert!(
            infer.len() >= self.needed_input(),
            "SOLA input too short: {} < {}",
            infer.len(),
            self.needed_input()
        );
        // normalised cross-correlation of the buffer against the first buf_len+search samples
        let mut best = 0usize;
        if self.primed && self.buf_len > 0 {
            let mut best_v = f32::NEG_INFINITY;
            for lag in 0..=self.search {
                let mut nom = 0.0f32;
                let mut den = 0.0f32;
                for i in 0..self.buf_len {
                    let s = infer[lag + i];
                    nom += s * self.buffer[i];
                    den += s * s;
                }
                let v = nom / (den + 1e-8).sqrt();
                if v > best_v {
                    best_v = v;
                    best = lag;
                }
            }
        }
        let seg = &infer[best..];
        let mut out = Vec::with_capacity(self.block);
        for i in 0..self.block {
            let v = if i < self.buf_len {
                seg[i] * self.fade_in[i] + self.buffer[i] * self.fade_out[i]
            } else {
                seg[i]
            };
            out.push(v);
        }
        // the next cross-fade starts where this block ended
        self.buffer
            .copy_from_slice(&seg[self.block..self.block + self.buf_len]);
        self.primed = true;
        out
    }
}
