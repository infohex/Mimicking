//! Energy-based voice activity gate with an adaptive noise floor and hysteresis.
//!
//! A diffusion-based converter "invents" speech in silence, so the output must be gated. This gate
//! works on the raw input blocks (before the model) and returns 0/1 per block; the pipeline ramps the
//! output over one block on transitions and applies the decision to the output it belongs to (the block
//! emitted now carries input from the right context ago).
//!
//! Opening needs *sustained* energy: three of the block's four quarters above the threshold, or the last
//! two (speech that starts in the second half of a block). A keyboard click or a cough fills one quarter
//! and is ignored; speech fills the block. Without this the gate opened on room transients and the model
//! produced voice-like fragments out of noise.

use mimic_core::Vad;

#[derive(Debug, Clone)]
pub struct EnergyVad {
    /// below this absolute level (dBFS) the block is always silence
    pub abs_floor_db: f32,
    /// open when the block is this many dB above the tracked floor
    pub open_db: f32,
    /// close when the block drops within this many dB of the floor ...
    pub close_db: f32,
    /// ... for this many consecutive blocks (2 blocks of 0.25 s rides over pauses between phrases; 1 cut
    /// speech at every breath)
    pub hold_blocks: usize,
    floor_db: Option<f32>,
    open: bool,
    quiet: usize,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self {
            abs_floor_db: -60.0,
            open_db: 12.0,
            close_db: 6.0,
            hold_blocks: 2,
            floor_db: None,
            open: false,
            quiet: 0,
        }
    }
}

impl EnergyVad {
    pub fn new(abs_floor_db: f32, hold_blocks: usize) -> Self {
        Self {
            abs_floor_db,
            hold_blocks,
            ..Default::default()
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }
}

impl Vad for EnergyVad {
    fn process(&mut self, block: &[f32]) -> f32 {
        let level = |x: &[f32]| {
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
            20.0 * (rms + 1e-9).log10()
        };
        let db = level(block);
        let q = (block.len() / 4).max(1);
        let quarters: Vec<f32> = block.chunks(q).take(4).map(level).collect();
        let floor = self.floor_db.get_or_insert(db);
        // the floor follows drops immediately and rises slowly (slower while speech is on)
        let rise = if self.open { 0.1 } else { 1.0 };
        *floor = if db > *floor {
            (*floor + rise).min(db)
        } else {
            db
        };
        let floor = *floor;
        if db < self.abs_floor_db {
            self.open = false;
            self.quiet = 0;
            return 0.0;
        }
        if !self.open {
            let hot: Vec<bool> = quarters.iter().map(|&v| v > floor + self.open_db).collect();
            let loud = hot.iter().filter(|&&h| h).count();
            let tail = hot.len() >= 4 && hot[2] && hot[3];
            if loud >= 3 || tail {
                self.open = true;
                self.quiet = 0;
            }
        } else if db < floor + self.close_db {
            self.quiet += 1;
            if self.quiet >= self.hold_blocks {
                self.open = false;
            }
        } else {
            self.quiet = 0;
        }
        if self.open {
            1.0
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.floor_db = None;
        self.open = false;
        self.quiet = 0;
    }
}
