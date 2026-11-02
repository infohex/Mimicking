//! Silero VAD (github.com/snakers4/silero-vad, MIT) as a block gate.
//!
//! The energy gate in `mimic-dsp` only sees loudness, so continuous typing, desk taps and coughs open it as
//! readily as speech (measured on real recordings: 75–95 % of blocks), and once open the converter turns the
//! noise into full-level babble. Silero is a small recurrent network trained on thousands of hours of speech and
//! noise in many languages: it answers "is somebody speaking in these 32 ms" with a probability, and on the same
//! recordings it kept keyboard and desk noise at 0 blocks while recalling essentially every speech block. It
//! runs on the CPU, 0.15 ms per 32 ms chunk, with ONNX Runtime's existing environment (one per process).
//!
//! Model protocol (v5/v6 ONNX, 16 kHz): each call takes the previous chunk's last 64 samples as context followed
//! by 512 new samples, the recurrent `state` (2, 1, 128) is carried between calls, `sr` is a scalar int64.
//! The block decision follows the energy gate's conventions: a block opens when any of its chunks exceeds
//! `threshold`, closes after `hold_blocks` consecutive blocks below 0.7 × threshold (hysteresis rides over pauses
//! between phrases), and a block under `abs_floor_db` is always silence. Like the energy gate it is a
//! speech / non-speech decision: other people's speech in the background passes by design.

use crate::{ort_err, OnnxRuntime};
use mimic_core::{Error, Result, Vad};
use ndarray::{s, Array2, Array3, ArrayView1, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const CHUNK: usize = 512;
const CTX: usize = 64;
const SR: i64 = 16000;

pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    context: [f32; CTX],
    /// 16 kHz samples that did not fill a whole chunk yet (a 240 ms block is 7.5 chunks)
    pending: Vec<f32>,
    /// chunk probability above which a block opens
    pub threshold: f32,
    /// consecutive quiet blocks before the gate closes
    pub hold_blocks: usize,
    /// below this block level (dBFS) the block is silence without asking the model
    pub abs_floor_db: f32,
    open: bool,
    quiet: usize,
    last_max: f32,
}

impl SileroVad {
    /// `path`: the 16 kHz Silero VAD graph (`silero_vad_16k_op15.onnx` of the v6 release, 1.2 MB). The session
    /// runs on the CPU whatever device `rt` targets.
    pub fn load(rt: &OnnxRuntime, path: &Path) -> Result<Self> {
        let session = Session::builder(&rt.env)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .with_config_entry("session.intra_op.allow_spinning", "0")
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(|e| Error::Backend(format!("loading {}: {e}", path.display())))?;
        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        for need in ["input", "state", "sr"] {
            if !names.iter().any(|n| n == need) {
                return Err(Error::Backend(format!("{}: not a Silero VAD graph (inputs {names:?})", path.display())));
            }
        }
        Ok(Self {
            session,
            state: Array3::zeros((2, 1, 128)),
            context: [0.0; CTX],
            pending: Vec::with_capacity(2 * CHUNK),
            threshold: 0.5,
            hold_blocks: 2,
            abs_floor_db: -60.0,
            open: false,
            quiet: 0,
            last_max: 0.0,
        })
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Highest chunk probability of the last block.
    pub fn last_prob(&self) -> f32 {
        self.last_max
    }

    fn chunk_prob(&mut self, chunk: &[f32]) -> Result<f32> {
        let mut input = Array2::<f32>::zeros((1, CTX + CHUNK));
        input.slice_mut(s![0, ..CTX]).assign(&ArrayView1::from(&self.context[..]));
        input.slice_mut(s![0, CTX..]).assign(&ArrayView1::from(chunk));
        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("input".to_string(), Tensor::from_array(input.into_dyn()).map_err(ort_err)?.into_dyn()),
            ("state".to_string(), Tensor::from_array(self.state.clone().into_dyn()).map_err(ort_err)?.into_dyn()),
            ("sr".to_string(), Tensor::from_array(ndarray::arr0(SR).into_dyn()).map_err(ort_err)?.into_dyn()),
        ];
        let out = self.session.run(inputs).map_err(ort_err)?;
        let p = out
            .get("output")
            .ok_or_else(|| Error::Backend("silero: missing output".into()))?
            .try_extract_array::<f32>()
            .map_err(ort_err)?
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);
        let st = out
            .get("stateN")
            .ok_or_else(|| Error::Backend("silero: missing stateN".into()))?
            .try_extract_array::<f32>()
            .map_err(ort_err)?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(|e| Error::Shape(format!("silero state: {e}")))?;
        self.state = st;
        self.context.copy_from_slice(&chunk[CHUNK - CTX..]);
        Ok(p)
    }
}

impl Vad for SileroVad {
    fn process(&mut self, block: &[f32]) -> f32 {
        let rms = (block.iter().map(|v| v * v).sum::<f32>() / block.len().max(1) as f32).sqrt();
        let db = 20.0 * (rms + 1e-9).log10();
        self.pending.extend_from_slice(block);
        let mut mx = 0.0f32;
        let mut off = 0;
        while self.pending.len() - off >= CHUNK {
            let chunk: Vec<f32> = self.pending[off..off + CHUNK].to_vec();
            match self.chunk_prob(&chunk) {
                Ok(p) => mx = mx.max(p),
                Err(e) => tracing::warn!(error = %e, "silero vad chunk failed"),
            }
            off += CHUNK;
        }
        self.pending.drain(..off);
        self.last_max = mx;
        if db < self.abs_floor_db {
            self.open = false;
            self.quiet = 0;
            return 0.0;
        }
        if !self.open {
            if mx > self.threshold {
                self.open = true;
                self.quiet = 0;
            }
        } else if mx < 0.7 * self.threshold {
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
        self.state.fill(0.0);
        self.context = [0.0; CTX];
        self.pending.clear();
        self.open = false;
        self.quiet = 0;
        self.last_max = 0.0;
    }
}
