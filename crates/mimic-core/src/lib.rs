//! Stage boundaries of a zero-shot voice-conversion pipeline.
//!
//! Every stage is a trait so that model packages (e.g. `mimic-model`) and inference backends
//! (e.g. `mimic-onnx`) can be swapped without touching the pipeline. Nothing in this crate knows
//! about a particular model, file format, runtime, sample-rate convention of a consumer, or process boundary.
//!
//! Tensor layout conventions (all `f32`, batch-first):
//! - waveforms: `(B, N)` samples
//! - content features: `(B, T, D)` frames x channels
//! - conditions / regulated content: `(B, T, D)`
//! - mel spectrograms: `(B, n_mels, T)` (channel-major, as the acoustic models consume them)
//! - speaker embeddings: `(B, E)`

use ndarray::{Array1, Array2, Array3, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Executes a fixed computation graph (one ONNX model, one TensorRT engine, ...).
///
/// Inputs are given positionally in the graph's declared input order. Implementations may require
/// fixed shapes (static graphs); `input_shapes` reports them (`None` entries are dynamic).
pub trait Graph: Send {
    fn input_names(&self) -> &[String];
    fn output_names(&self) -> &[String];
    fn run(&mut self, inputs: &[ArrayViewD<'_, f32>]) -> Result<Vec<ArrayD<f32>>>;

    /// Set one input and keep it for subsequent [`run_set`](Self::run_set) calls. Lets a caller that re-runs a
    /// graph with mostly unchanged inputs (an ODE solver) avoid re-sending the unchanged ones.
    fn set_input(&mut self, index: usize, value: ArrayViewD<'_, f32>) -> Result<()>;

    /// Run with the inputs previously given to [`set_input`](Self::set_input).
    fn run_set(&mut self) -> Result<Vec<ArrayD<f32>>>;

    /// Run with the set inputs but leave the outputs where the backend keeps them (on the device for a
    /// bound graph); read them later with [`get_output`](Self::get_output).
    fn run_set_keep(&mut self) -> Result<()>;

    /// Fetch output `index` of the last run.
    fn get_output(&mut self, index: usize) -> Result<ArrayD<f32>>;

    /// Copy output `output` of the last run onto input `input` without going through the host
    /// (device-to-device for a bound graph). Shapes must match.
    fn copy_output_to_input(&mut self, output: usize, input: usize) -> Result<()>;
}

/// A velocity estimator that also applies the ODE update itself and keeps the state on its own side
/// (e.g. a graph with the Euler/CFG update fused in and `x` resident on the device).
///
/// Batch layout as for [`VelocityEstimator`]: row 0 conditional, row 1 unconditional; both rows carry the
/// same `x` and receive the same updated `x`.
pub trait FlowStepper: Send {
    /// Load the initial state `x` `(2, n_mels, L)` and the per-utterance context.
    fn begin(
        &mut self,
        x: ArrayView3<'_, f32>,
        prompt_x: ArrayView3<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<()>;
    /// `x <- x + dt * ((1 + cfg) v_cond(x, t) - cfg v_uncond(x, t))`, prompt frames kept at zero.
    fn step(&mut self, t: f32, dt: f32, cfg: f32) -> Result<()>;
    /// Current state `(2, n_mels, L)`.
    fn state(&mut self) -> Result<Array3<f32>>;
    fn n_mels(&self) -> usize;
}

/// Speech -> frame-rate content (linguistic) features. Input is 16 kHz audio.
pub trait ContentEncoder {
    /// Frame-rate content features for a batch of equal-length 16 kHz waveforms: `(B, N) -> (B, T, D)`.
    fn encode(&mut self, wav16k: ArrayView2<'_, f32>) -> Result<Array3<f32>>;
    /// Number of output frames produced for `n_samples` input samples.
    fn frames_for(&self, n_samples: usize) -> usize;
    fn feature_dim(&self) -> usize;
}

/// Reference speech -> one global timbre embedding `(E,)`. Input is 16 kHz audio of any length.
pub trait SpeakerEncoder {
    fn embed(&mut self, wav16k: &[f32]) -> Result<Array1<f32>>;
    fn embedding_dim(&self) -> usize;
}

/// Content features at their own frame rate -> acoustic-frame-rate condition sequence.
pub trait LengthRegulator {
    fn regulate(
        &mut self,
        content: ArrayView3<'_, f32>,
        target_frames: usize,
    ) -> Result<Array3<f32>>;
    fn out_dim(&self) -> usize;
}

/// One evaluation of the flow-matching vector field. The ODE solver lives outside (see `mimic-pipeline`).
///
/// `x`/`prompt_x`: `(B, n_mels, L)`; `t`: `(B,)`; `style`: `(B, E)`; `mu`: `(B, L, D)`. Returns `(B, n_mels, L)`.
/// Classifier-free guidance is expressed by the caller through the batch (conditional rows followed by
/// unconditional rows), so the estimator stays a plain function.
pub trait VelocityEstimator {
    fn velocity(
        &mut self,
        x: ArrayView3<'_, f32>,
        prompt_x: ArrayView3<'_, f32>,
        t: ArrayView1<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<Array3<f32>>;
    fn n_mels(&self) -> usize;

    /// Fix the per-utterance inputs (`prompt_x`, `style`, `mu`) for a run of [`velocity_step`](Self::velocity_step)
    /// calls. Default: remember them and fall back to [`velocity`](Self::velocity).
    fn set_context(
        &mut self,
        prompt_x: ArrayView3<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<()>;

    /// One velocity evaluation with the context given to [`set_context`](Self::set_context).
    fn velocity_step(
        &mut self,
        x: ArrayView3<'_, f32>,
        t: ArrayView1<'_, f32>,
    ) -> Result<Array3<f32>>;
}

/// Mel spectrogram `(B, n_mels, T)` -> waveform `(B, L)` at the vocoder's sample rate.
pub trait Vocoder {
    fn vocode(&mut self, mel: ArrayView3<'_, f32>) -> Result<Array2<f32>>;
    fn sample_rate(&self) -> u32;
    /// Output samples per mel frame.
    fn hop(&self) -> usize;
}

/// Waveform -> mel spectrogram `(n_mels, T)` in the convention the acoustic model was trained with.
pub trait MelAnalyzer {
    fn analyze(&self, wav: &[f32]) -> Array2<f32>;
    fn n_mels(&self) -> usize;
    fn hop(&self) -> usize;
    fn sample_rate(&self) -> u32;
    /// Number of mel frames for `n_samples` input samples.
    fn frames_for(&self, n_samples: usize) -> usize;
}

/// Voice-activity gate over consecutive input blocks: returns the gate value for this block (1 = speech, 0 = silence).
pub trait Vad: Send {
    fn process(&mut self, block: &[f32]) -> f32;
    fn reset(&mut self);
}

/// Sample-rate conversion of a whole signal (offline) with a fixed ratio.
pub trait Resampler {
    fn resample(&self, x: &[f32]) -> Vec<f32>;
    fn output_len(&self, n_samples: usize) -> usize;
}

/// Everything derived from one reference utterance, computed once and reused for every converted chunk.
#[derive(Debug, Clone)]
pub struct Prompt {
    /// Regulated content condition of the reference `(1, P, D)`.
    pub cond: Array3<f32>,
    /// Reference mel spectrogram `(1, n_mels, P)`.
    pub mel: Array3<f32>,
    /// Speaker embedding `(1, E)`.
    pub style: Array2<f32>,
}

impl Prompt {
    pub fn frames(&self) -> usize {
        self.mel.shape()[2]
    }
}
