//! Model package for the XLS-R → length regulator → DiT → HiFT family (the Mimicking 2.0 self-trained sets and the
//! seed-vc tiny baseline sets `onnx*` load through the same code): XLSR-300m (12 layers) content encoder,
//! CAM++ speaker encoder, interpolating length regulator, 25M-parameter DiT flow-matching estimator and the
//! HiFT vocoder, all served from the static ONNX graphs produced by MimickingTrain's `mimictrain.export`.
//!
//! The crate implements the `mimic-core` stage traits; the pipeline that strings them together lives in
//! `mimic-pipeline`. Shapes are fixed by [`WindowConfig`] (block length, context lengths, prompt length) and
//! must match the graphs that were exported for that configuration.

use mimic_core::{
    ContentEncoder, Error, Graph, LengthRegulator, MelAnalyzer, Result, SpeakerEncoder,
    VelocityEstimator, Vocoder,
};
use mimic_dsp::hift::{harmonic_source, istft_from_body, stft_small, SourceParams, StftParams};
use mimic_dsp::{KaldiFbank, MelSpec, SincResampler};
use mimic_onnx::{BoundGraph, OnnxGraph};
pub use mimic_onnx::{Device, OnnxRuntime, SileroVad};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Ix1, Ix2, Ix3, IxDyn};
use ndarray_npy::NpzReader;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::File;
use std::path::{Path, PathBuf};

pub const SR: usize = 22050;
pub const HOP: usize = 256;
pub const N_MELS: usize = 80;
pub const CONTENT_SR: usize = 16000;
pub const CONTENT_DIM: usize = 1024;
pub const COND_DIM: usize = 384;
pub const STYLE_DIM: usize = 192;

/// Real-time window configuration (mirrors seed-vc's `real-time-gui.py` arithmetic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowConfig {
    pub block_time: f32,
    pub crossfade_time: f32,
    pub extra_time_ce: f32,
    pub extra_time: f32,
    pub extra_time_right: f32,
    pub prompt_len_s: f32,
}

impl WindowConfig {
    /// Window configuration recorded in a model directory's `manifest.json`; fields the manifest lacks keep
    /// their defaults.
    pub fn from_manifest(m: &Manifest) -> Self {
        let mut c = Self::default();
        let f = |k: &str| m.config.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
        if let Some(v) = f("block_time") {
            c.block_time = v;
        }
        if let Some(v) = f("crossfade_time") {
            c.crossfade_time = v;
        }
        if let Some(v) = f("extra_time_ce") {
            c.extra_time_ce = v;
        }
        if let Some(v) = f("extra_time") {
            c.extra_time = v;
        }
        if let Some(v) = f("extra_time_right") {
            c.extra_time_right = v;
        }
        if let Some(v) = f("prompt_len_s") {
            c.prompt_len_s = v;
        }
        c
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            block_time: 0.25,
            crossfade_time: 0.05,
            extra_time_ce: 2.5,
            extra_time: 0.5,
            extra_time_right: 2.0,
            prompt_len_s: 3.0,
        }
    }
}

/// Every derived size, in samples (22.05 kHz unless stated) and frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shapes {
    pub zc: usize,
    pub block_frame: usize,
    pub crossfade_frame: usize,
    pub sola_buffer_frame: usize,
    pub sola_search_frame: usize,
    pub extra_frame: usize,
    pub extra_frame_right: usize,
    /// analysis window at 22.05 kHz
    pub window: usize,
    /// analysis window at 16 kHz (content-encoder input)
    pub window_16k: usize,
    /// new 16 kHz samples per hop
    pub block_16k: usize,
    pub skip_head: usize,
    pub skip_tail: usize,
    pub return_length: usize,
    /// content frames (50 Hz) dropped at the start before length regulation
    pub ce_dit_frame_difference: usize,
    /// acoustic frames the DiT generates for one block
    pub t_cond: usize,
    /// reference prompt samples (22.05 kHz) and mel frames
    pub prompt_samples: usize,
    pub prompt_frames: usize,
    pub seq_len: usize,
}

impl WindowConfig {
    pub fn shapes(&self) -> Shapes {
        let sr = SR as f32;
        let zc = SR / 50;
        // numpy.round rounds half to even (0.25 s -> 12.5 -> 12 units), so mirror that exactly
        let rnd = |t: f32| ((t * sr / zc as f32).round_ties_even() as usize) * zc;
        let block_frame = rnd(self.block_time);
        let crossfade_frame = rnd(self.crossfade_time);
        let sola_buffer_frame = crossfade_frame.min(4 * zc);
        let sola_search_frame = zc;
        let extra_frame = rnd(self.extra_time_ce);
        let extra_frame_right = rnd(self.extra_time_right);
        let window =
            extra_frame + crossfade_frame + sola_search_frame + block_frame + extra_frame_right;
        let window_16k = 320 * window / zc;
        let block_16k = 320 * block_frame / zc;
        let skip_head = extra_frame / zc;
        let skip_tail = extra_frame_right / zc;
        let return_length = (block_frame + sola_buffer_frame + sola_search_frame) / zc;
        let ce_dit_frame_difference = ((self.extra_time_ce - self.extra_time) * 50.0) as usize;
        // python: int((skip_head + return_length + skip_tail - ce) / 50 * sr // hop)
        let t_cond = (((skip_head + return_length + skip_tail - ce_dit_frame_difference) as f64
            / 50.0
            * SR as f64)
            / HOP as f64)
            .floor() as usize;
        let prompt_samples = (sr * self.prompt_len_s) as usize;
        let prompt_frames = (prompt_samples - HOP) / HOP + 1;
        Shapes {
            zc,
            block_frame,
            crossfade_frame,
            sola_buffer_frame,
            sola_search_frame,
            extra_frame,
            extra_frame_right,
            window,
            window_16k,
            block_16k,
            skip_head,
            skip_tail,
            return_length,
            ce_dit_frame_difference,
            t_cond,
            prompt_samples,
            prompt_frames,
            seq_len: prompt_frames + t_cond,
        }
    }
}

/// File names inside the models directory.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub dir: PathBuf,
}

impl ModelFiles {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
    pub fn xlsr(&self) -> PathBuf {
        self.dir.join("xlsr.onnx")
    }
    pub fn xlsr_ref(&self) -> PathBuf {
        self.dir.join("xlsr_ref.onnx")
    }
    pub fn length_regulator(&self) -> PathBuf {
        self.dir.join("length_regulator.onnx")
    }
    pub fn length_regulator_ref(&self) -> PathBuf {
        self.dir.join("length_regulator_ref.onnx")
    }
    pub fn dit_step(&self) -> PathBuf {
        self.dir.join("dit_step.onnx")
    }
    pub fn dit_step_fused(&self) -> PathBuf {
        self.dir.join("dit_step_fused.onnx")
    }
    pub fn campplus(&self) -> PathBuf {
        self.dir.join("campplus.onnx")
    }
    pub fn hift_f0(&self) -> PathBuf {
        self.dir.join("hift_f0.onnx")
    }
    pub fn hift_body(&self) -> PathBuf {
        self.dir.join("hift_body.onnx")
    }
    pub fn hift_source(&self) -> PathBuf {
        self.dir.join("hift_source.npz")
    }
}

fn dyn_to<D: ndarray::Dimension>(
    a: ndarray::ArrayD<f32>,
    what: &str,
) -> Result<ndarray::Array<f32, D>> {
    a.into_dimensionality::<D>()
        .map_err(|e| Error::Shape(format!("{what}: {e}")))
}

/// Static-shape graph: device-resident bound session (CUDA Graph capable) on CUDA, plain session on CPU.
fn load_static(
    rt: &OnnxRuntime,
    path: &Path,
    ins: &[Vec<usize>],
    outs: &[Vec<usize>],
) -> Result<Box<dyn Graph>> {
    let hint = |e: Error| {
        Error::Backend(format!(
            "{e}\n  (shapes requested {ins:?} -> {outs:?}; the graphs in {} must be exported for this block/context configuration: \
             export a set for this geometry: python -m mimictrain.export --block <seconds> --right <seconds>)",
            path.parent().map(|p| p.display().to_string()).unwrap_or_default()
        ))
    };
    match rt.device() {
        Device::Cuda(_) => {
            let mut g = BoundGraph::load(rt, path, ins, outs).map_err(hint)?;
            g.warm_up().map_err(hint)?;
            Ok(Box::new(g))
        }
        Device::Cpu => Ok(Box::new(rt.load_graph(path)?)),
    }
}

/// HF wav2vec2 feature-encoder frame count: conv stack (10,5), (3,2) x4, (2,2) x2.
pub fn xlsr_frames_for(n_samples: usize) -> usize {
    let mut l = n_samples;
    for (k, s) in [
        (10usize, 5usize),
        (3, 2),
        (3, 2),
        (3, 2),
        (3, 2),
        (2, 2),
        (2, 2),
    ] {
        l = (l - k) / s + 1;
    }
    l
}

// ------------------------------------------------------------------------------------------------ XLSR

/// wav2vec2 XLS-R 300m, first 12 transformer layers, fp16 inside the graph. Fixed input length.
pub struct XlsrEncoder {
    graph: Box<dyn Graph>,
    input_len: usize,
}

impl XlsrEncoder {
    pub fn load(rt: &OnnxRuntime, path: &Path, input_len: usize) -> Result<Self> {
        let frames = xlsr_frames_for(input_len);
        let graph = load_static(
            rt,
            path,
            &[vec![1, input_len]],
            &[vec![1, frames, CONTENT_DIM]],
        )?;
        Ok(Self { graph, input_len })
    }
    pub fn input_len(&self) -> usize {
        self.input_len
    }
}

impl ContentEncoder for XlsrEncoder {
    fn encode(&mut self, wav16k: ArrayView2<'_, f32>) -> Result<Array3<f32>> {
        if wav16k.shape()[1] != self.input_len {
            return Err(Error::Shape(format!(
                "xlsr expects {} samples, got {}",
                self.input_len,
                wav16k.shape()[1]
            )));
        }
        let out = self.graph.run(&[wav16k.into_dyn()])?.remove(0);
        dyn_to::<Ix3>(out, "xlsr output")
    }
    fn frames_for(&self, n_samples: usize) -> usize {
        xlsr_frames_for(n_samples)
    }
    fn feature_dim(&self) -> usize {
        CONTENT_DIM
    }
}

// ------------------------------------------------------------------------------------------------ length regulator

pub struct InterpolateRegulator {
    graph: Box<dyn Graph>,
    in_frames: usize,
    out_frames: usize,
}

impl InterpolateRegulator {
    pub fn load(
        rt: &OnnxRuntime,
        path: &Path,
        in_frames: usize,
        out_frames: usize,
    ) -> Result<Self> {
        let graph = load_static(
            rt,
            path,
            &[vec![1, in_frames, CONTENT_DIM]],
            &[vec![1, out_frames, COND_DIM]],
        )?;
        Ok(Self {
            graph,
            in_frames,
            out_frames,
        })
    }
}

impl LengthRegulator for InterpolateRegulator {
    fn regulate(
        &mut self,
        content: ArrayView3<'_, f32>,
        target_frames: usize,
    ) -> Result<Array3<f32>> {
        if target_frames != self.out_frames || content.shape()[1] != self.in_frames {
            return Err(Error::Shape(format!(
                "regulator graph is fixed to {} -> {} frames, asked {} -> {}",
                self.in_frames,
                self.out_frames,
                content.shape()[1],
                target_frames
            )));
        }
        let out = self.graph.run(&[content.into_dyn()])?.remove(0);
        dyn_to::<Ix3>(out, "regulator output")
    }
    fn out_dim(&self) -> usize {
        COND_DIM
    }
}

// ------------------------------------------------------------------------------------------------ CAM++

pub struct CampplusSpeaker {
    graph: OnnxGraph,
    fbank: KaldiFbank,
}

impl CampplusSpeaker {
    pub fn load(rt: &OnnxRuntime, path: &Path) -> Result<Self> {
        Ok(Self {
            graph: rt.load_graph(path)?,
            fbank: KaldiFbank::campplus_16k(),
        })
    }
}

impl SpeakerEncoder for CampplusSpeaker {
    fn embed(&mut self, wav16k: &[f32]) -> Result<Array1<f32>> {
        let f = self.fbank.compute_mean_normalized(wav16k); // (T, 80)
        let (t, b) = f.dim();
        let f3 = f
            .into_shape_with_order((1, t, b))
            .map_err(|e| Error::Shape(e.to_string()))?;
        let out = self.graph.run(&[f3.view().into_dyn()])?.remove(0);
        let out = dyn_to::<Ix2>(out, "campplus output")?;
        Ok(out.row(0).to_owned())
    }
    fn embedding_dim(&self) -> usize {
        STYLE_DIM
    }
}

// ------------------------------------------------------------------------------------------------ DiT step

pub struct DitStep {
    graph: Box<dyn Graph>,
    batch: usize,
    seq_len: usize,
}

impl DitStep {
    pub fn load(rt: &OnnxRuntime, path: &Path, batch: usize, seq_len: usize) -> Result<Self> {
        let (b, l) = (batch, seq_len);
        let graph = load_static(
            rt,
            path,
            &[
                vec![b, N_MELS, l],
                vec![b, N_MELS, l],
                vec![b],
                vec![b, STYLE_DIM],
                vec![b, l, COND_DIM],
            ],
            &[vec![b, N_MELS, l]],
        )?;
        Ok(Self {
            graph,
            batch,
            seq_len,
        })
    }
    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

impl VelocityEstimator for DitStep {
    fn velocity(
        &mut self,
        x: ArrayView3<'_, f32>,
        prompt_x: ArrayView3<'_, f32>,
        t: ArrayView1<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<Array3<f32>> {
        if x.shape() != [self.batch, N_MELS, self.seq_len] {
            return Err(Error::Shape(format!(
                "dit_step expects x {:?}, got {:?}",
                [self.batch, N_MELS, self.seq_len],
                x.shape()
            )));
        }
        let out = self
            .graph
            .run(&[
                x.into_dyn(),
                prompt_x.into_dyn(),
                t.into_dyn(),
                style.into_dyn(),
                mu.into_dyn(),
            ])?
            .remove(0);
        dyn_to::<Ix3>(out, "dit output")
    }
    fn n_mels(&self) -> usize {
        N_MELS
    }
    fn set_context(
        &mut self,
        prompt_x: ArrayView3<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<()> {
        self.graph.set_input(1, prompt_x.into_dyn())?;
        self.graph.set_input(3, style.into_dyn())?;
        self.graph.set_input(4, mu.into_dyn())
    }
    fn velocity_step(
        &mut self,
        x: ArrayView3<'_, f32>,
        t: ArrayView1<'_, f32>,
    ) -> Result<Array3<f32>> {
        self.graph.set_input(0, x.into_dyn())?;
        self.graph.set_input(2, t.into_dyn())?;
        let out = self.graph.run_set()?.remove(0);
        dyn_to::<Ix3>(out, "dit output")
    }
}

// ------------------------------------------------------------------------------------------------ DiT fused step

/// DiT step with the Euler/CFG update fused into the graph; `x` stays on the device between steps.
/// Inputs: x, prompt_x, t, style, mu, dt, cfg -> x_next.
pub struct DitStepFused {
    graph: Box<dyn Graph>,
    batch: usize,
    seq_len: usize,
}

impl DitStepFused {
    pub fn load(rt: &OnnxRuntime, path: &Path, batch: usize, seq_len: usize) -> Result<Self> {
        let (b, l) = (batch, seq_len);
        let graph = load_static(
            rt,
            path,
            &[
                vec![b, N_MELS, l],
                vec![b, N_MELS, l],
                vec![b],
                vec![b, STYLE_DIM],
                vec![b, l, COND_DIM],
                vec![1],
                vec![1],
            ],
            &[vec![b, N_MELS, l]],
        )?;
        Ok(Self {
            graph,
            batch,
            seq_len,
        })
    }
}

impl mimic_core::FlowStepper for DitStepFused {
    fn begin(
        &mut self,
        x: ArrayView3<'_, f32>,
        prompt_x: ArrayView3<'_, f32>,
        style: ArrayView2<'_, f32>,
        mu: ArrayView3<'_, f32>,
    ) -> Result<()> {
        if x.shape() != [self.batch, N_MELS, self.seq_len] {
            return Err(Error::Shape(format!(
                "fused step expects x {:?}, got {:?}",
                [self.batch, N_MELS, self.seq_len],
                x.shape()
            )));
        }
        self.graph.set_input(0, x.into_dyn())?;
        self.graph.set_input(1, prompt_x.into_dyn())?;
        self.graph.set_input(3, style.into_dyn())?;
        self.graph.set_input(4, mu.into_dyn())
    }
    fn step(&mut self, t: f32, dt: f32, cfg: f32) -> Result<()> {
        let tt = Array1::from_vec(vec![t; self.batch]);
        self.graph.set_input(2, tt.view().into_dyn())?;
        self.graph
            .set_input(5, Array1::from_vec(vec![dt]).view().into_dyn())?;
        self.graph
            .set_input(6, Array1::from_vec(vec![cfg]).view().into_dyn())?;
        self.graph.run_set_keep()?;
        self.graph.copy_output_to_input(0, 0)
    }
    fn state(&mut self) -> Result<Array3<f32>> {
        dyn_to::<Ix3>(self.graph.get_output(0)?, "fused step state")
    }
    fn n_mels(&self) -> usize {
        N_MELS
    }
}

// ------------------------------------------------------------------------------------------------ HiFT

pub struct HiftVocoder {
    f0: Box<dyn Graph>,
    body: Box<dyn Graph>,
    source: SourceParams,
    stft: StftParams,
    rng: StdRng,
    mel_frames: usize,
}

fn npz_f32(r: &mut NpzReader<File>, name: &str) -> Result<ndarray::ArrayD<f32>> {
    r.by_name::<ndarray::OwnedRepr<f32>, IxDyn>(&format!("{name}.npy"))
        .map_err(|e| Error::Other(format!("hift_source.npz/{name}: {e}")))
}
/// numpy 0-d scalars: int64 (Linux) or int32 (Windows numpy 1.x) for ints, float64 for floats.
fn npz_scalar(r: &mut NpzReader<File>, name: &str) -> Result<f64> {
    let n = format!("{name}.npy");
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<i64>, IxDyn>(&n) {
        return a
            .iter()
            .next()
            .map(|v| *v as f64)
            .ok_or_else(|| Error::Other(format!("{name} empty")));
    }
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<i32>, IxDyn>(&n) {
        return a
            .iter()
            .next()
            .map(|v| *v as f64)
            .ok_or_else(|| Error::Other(format!("{name} empty")));
    }
    if let Ok(a) = r.by_name::<ndarray::OwnedRepr<f64>, IxDyn>(&n) {
        return a
            .iter()
            .next()
            .copied()
            .ok_or_else(|| Error::Other(format!("{name} empty")));
    }
    let a = r
        .by_name::<ndarray::OwnedRepr<f32>, IxDyn>(&n)
        .map_err(|e| Error::Other(format!("hift_source.npz/{name}: {e}")))?;
    a.iter()
        .next()
        .map(|v| *v as f64)
        .ok_or_else(|| Error::Other(format!("{name} empty")))
}
fn npz_scalar_i64(r: &mut NpzReader<File>, name: &str) -> Result<i64> {
    Ok(npz_scalar(r, name)? as i64)
}
fn npz_scalar_f64(r: &mut NpzReader<File>, name: &str) -> Result<f64> {
    npz_scalar(r, name)
}

impl HiftVocoder {
    /// `mel_frames`: the fixed number of mel frames per call (the block's `t_cond`).
    pub fn load(rt: &OnnxRuntime, files: &ModelFiles, mel_frames: usize) -> Result<Self> {
        let mut r = NpzReader::new(File::open(files.hift_source())?)
            .map_err(|e| Error::Other(e.to_string()))?;
        let w = npz_f32(&mut r, "l_linear_weight")?;
        let w = dyn_to::<Ix2>(w, "l_linear_weight")?;
        let b = dyn_to::<Ix1>(npz_f32(&mut r, "l_linear_bias")?, "l_linear_bias")?;
        let window = dyn_to::<Ix1>(npz_f32(&mut r, "window")?, "window")?.to_vec();
        let source = SourceParams {
            sampling_rate: npz_scalar_i64(&mut r, "sampling_rate")? as f32,
            harmonic_num: npz_scalar_i64(&mut r, "harmonic_num")? as usize,
            sine_amp: npz_scalar_f64(&mut r, "sine_amp")? as f32,
            noise_std: npz_scalar_f64(&mut r, "noise_std")? as f32,
            voiced_threshold: npz_scalar_i64(&mut r, "voiced_threshold")? as f32,
            upsample: npz_scalar_i64(&mut r, "upsample")? as usize,
            linear_weight: w.row(0).to_vec(),
            linear_bias: b[0],
        };
        let stft = StftParams {
            n_fft: npz_scalar_i64(&mut r, "n_fft")? as usize,
            hop: npz_scalar_i64(&mut r, "hop_len")? as usize,
            window,
            audio_limit: npz_scalar_f64(&mut r, "audio_limit")? as f32,
        };
        let t = mel_frames;
        let tt = t * source.upsample / stft.hop + 1; // STFT frames of the upsampled source (center=True)
        let f0 = load_static(rt, &files.hift_f0(), &[vec![1, N_MELS, t]], &[vec![1, t]])?;
        let body = load_static(
            rt,
            &files.hift_body(),
            &[vec![1, N_MELS, t], vec![1, stft.n_fft + 2, tt]],
            &[vec![1, stft.n_fft + 2, tt]],
        )?;
        Ok(Self {
            f0,
            body,
            source,
            stft,
            rng: StdRng::seed_from_u64(0x5eed),
            mel_frames,
        })
    }

    pub fn mel_frames(&self) -> usize {
        self.mel_frames
    }

    pub fn source_params(&self) -> &SourceParams {
        &self.source
    }
    pub fn stft_params(&self) -> &StftParams {
        &self.stft
    }

    /// Vocode with an externally supplied excitation (used by parity tests to remove randomness).
    pub fn vocode_with_source(
        &mut self,
        mel: ArrayView3<'_, f32>,
        source: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        let t = mel.shape()[2];
        let s_stft = stft_small(source, &self.stft);
        let body = self
            .body
            .run(&[mel.into_dyn(), s_stft.view().into_dyn()])?
            .remove(0);
        let body = dyn_to::<Ix3>(body, "hift body output")?;
        Ok(istft_from_body(
            body.view(),
            &self.stft,
            t * self.source.upsample,
        ))
    }

    pub fn predict_f0(&mut self, mel: ArrayView3<'_, f32>) -> Result<Array2<f32>> {
        let f0 = self.f0.run(&[mel.into_dyn()])?.remove(0);
        dyn_to::<Ix2>(f0, "hift f0 output")
    }
}

impl Vocoder for HiftVocoder {
    fn vocode(&mut self, mel: ArrayView3<'_, f32>) -> Result<Array2<f32>> {
        let f0 = self.predict_f0(mel)?;
        let source = harmonic_source(f0.view(), &self.source, &mut self.rng);
        self.vocode_with_source(mel, source.view())
    }
    fn sample_rate(&self) -> u32 {
        SR as u32
    }
    fn hop(&self) -> usize {
        HOP
    }
}

// ------------------------------------------------------------------------------------------------ bundle

/// All stages of one model set for one window configuration.
pub struct ModelSet {
    pub config: WindowConfig,
    pub shapes: Shapes,
    pub encoder: XlsrEncoder,
    pub encoder_ref: XlsrEncoder,
    pub regulator: InterpolateRegulator,
    pub regulator_ref: InterpolateRegulator,
    pub speaker: CampplusSpeaker,
    pub estimator: DitStep,
    /// present when `dit_step_fused.onnx` exists next to the other graphs
    pub stepper: Option<DitStepFused>,
    pub vocoder: HiftVocoder,
    pub mel: MelSpec,
    pub to_16k: SincResampler,
}

impl ModelSet {
    pub fn load(rt: &OnnxRuntime, files: &ModelFiles, config: WindowConfig) -> Result<Self> {
        let sh = config.shapes();
        let encoder = XlsrEncoder::load(rt, &files.xlsr(), sh.window_16k)?;
        let ref_16k = (sh.prompt_samples * CONTENT_SR).div_ceil(SR); // ceil, as torchaudio.resample does
        let encoder_ref = XlsrEncoder::load(rt, &files.xlsr_ref(), ref_16k)?;
        let in_frames = encoder.frames_for(sh.window_16k) - sh.ce_dit_frame_difference;
        let regulator =
            InterpolateRegulator::load(rt, &files.length_regulator(), in_frames, sh.t_cond)?;
        let regulator_ref = InterpolateRegulator::load(
            rt,
            &files.length_regulator_ref(),
            encoder_ref.frames_for(ref_16k),
            sh.prompt_frames,
        )?;
        let speaker = CampplusSpeaker::load(rt, &files.campplus())?;
        let estimator = DitStep::load(rt, &files.dit_step(), 2, sh.seq_len)?;
        let stepper =
            if files.dit_step_fused().exists() && std::env::var_os("MIMIC_NO_FUSED").is_none() {
                Some(DitStepFused::load(
                    rt,
                    &files.dit_step_fused(),
                    2,
                    sh.seq_len,
                )?)
            } else {
                None
            };
        let vocoder = HiftVocoder::load(rt, files, sh.t_cond)?;
        let mel = MelSpec::mel_22k();
        debug_assert_eq!(mel.frames_for(sh.prompt_samples), sh.prompt_frames);
        Ok(Self {
            config,
            shapes: sh,
            encoder,
            encoder_ref,
            regulator,
            regulator_ref,
            speaker,
            estimator,
            stepper,
            vocoder,
            mel,
            to_16k: SincResampler::new(SR, CONTENT_SR),
        })
    }

    /// Streaming geometry of this configuration (what `mimic-pipeline` needs to drive the stages).
    pub fn geometry(&self) -> StreamGeometry {
        let sh = &self.shapes;
        StreamGeometry {
            acoustic_sr: SR,
            content_sr: CONTENT_SR,
            block_frame: sh.block_frame,
            sola_buffer_frame: sh.sola_buffer_frame,
            sola_search_frame: sh.sola_search_frame,
            extra_frame_right: sh.extra_frame_right,
            window_16k: sh.window_16k,
            block_16k: sh.block_16k,
            skip_tail: sh.skip_tail,
            return_length: sh.return_length,
            ce_dit_frame_difference: sh.ce_dit_frame_difference,
            t_cond: sh.t_cond,
            prompt_samples: sh.prompt_samples,
            prompt_frames: sh.prompt_frames,
        }
    }

    /// Hand the stages to `mimic-pipeline`. Uses the fused on-device Euler step when it was loaded.
    pub fn into_stages(self) -> (Stages, StreamGeometry) {
        let geo = self.geometry();
        let estimator = match self.stepper {
            Some(st) => Estimator::Fused(Box::new(st)),
            None => Estimator::Velocity(Box::new(self.estimator)),
        };
        let stages = Stages {
            content: Box::new(self.encoder),
            content_ref: Box::new(self.encoder_ref),
            regulator: Box::new(self.regulator),
            regulator_ref: Box::new(self.regulator_ref),
            speaker: Box::new(self.speaker),
            estimator,
            vocoder: Box::new(self.vocoder),
            mel: Box::new(self.mel),
            to_content_rate: Box::new(self.to_16k),
        };
        (stages, geo)
    }
}

// ------------------------------------------------------------------------------------------------ convenience re-exports

pub use mimic_pipeline::{Estimator, Stages, StreamGeometry};

// ------------------------------------------------------------------------------------------------ manifest

/// One entry of `manifest.json` (written by `mimictrain.export`).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ManifestEntry {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default)]
    pub role: String,
}

/// `manifest.json`: which graphs belong together, for which window configuration, built from which weights.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub config: serde_json::Value,
    pub files: Vec<ManifestEntry>,
    /// Where the weights and graphs came from: `provenance` in sets exported by `mimictrain.export`; the baseline
    /// sets' older manifests call the same block `upstream`.
    #[serde(default, alias = "upstream")]
    pub provenance: serde_json::Value,
}

impl ModelFiles {
    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    pub fn read_manifest(&self) -> Result<Manifest> {
        let text = std::fs::read_to_string(self.manifest_path())?;
        serde_json::from_str(&text).map_err(|e| Error::Other(format!("manifest.json: {e}")))
    }

    /// The window configuration the graphs in this directory were exported for, read from `manifest.json`.
    /// Falls back to [`WindowConfig::default`] when there is no manifest (or it carries no config), so a model
    /// directory can carry its own geometry (e.g. a low-latency set with a 0.02 s right context).
    pub fn config(&self) -> WindowConfig {
        self.read_manifest()
            .ok()
            .map(|m| WindowConfig::from_manifest(&m))
            .unwrap_or_default()
    }

    /// Check every file listed in the manifest: present, right size, right sha256. Returns the entries checked.
    pub fn verify(&self) -> Result<Vec<ManifestEntry>> {
        use sha2::{Digest, Sha256};
        let m = self.read_manifest()?;
        let mut bad = Vec::new();
        for e in &m.files {
            let p = self.dir.join(&e.file);
            let Ok(meta) = std::fs::metadata(&p) else {
                bad.push(format!("{}: missing", e.file));
                continue;
            };
            if meta.len() != e.bytes {
                bad.push(format!(
                    "{}: {} bytes, manifest says {}",
                    e.file,
                    meta.len(),
                    e.bytes
                ));
                continue;
            }
            let mut f = File::open(&p)?;
            let mut h = Sha256::new();
            std::io::copy(&mut f, &mut h)?;
            let got = format!("{:x}", h.finalize());
            if got != e.sha256 {
                bad.push(format!(
                    "{}: sha256 {got}, manifest says {}",
                    e.file, e.sha256
                ));
            }
        }
        if bad.is_empty() {
            Ok(m.files)
        } else {
            Err(Error::Other(format!(
                "model verification failed:\n  {}",
                bad.join("\n  ")
            )))
        }
    }
}
