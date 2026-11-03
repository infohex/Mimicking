//! Pipeline: strings the stage traits together.
//!
//! - [`EulerCfg`]: fixed-step Euler ODE solver with classifier-free guidance (the model's velocity field is a
//!   [`VelocityEstimator`], so the solver can be swapped without touching the model package).
//! - [`Stages`]: the set of trait objects one conversion needs, plus [`Stages::prepare_prompt`].
//! - [`Session`]: per-voice streaming state (analysis window, SOLA, gate, RNG, prompt). Owns no model.
//!   [`Session::roll`] advances that state through a quiet block without any model work.
//! - [`Runner`]: owns the [`Stages`] and advances any number of [`Session`]s one block at a time.
//! - [`Converter`]: whole-file conversion built on the same runner/session code path.

use mimic_core::{
    ContentEncoder, Error, FlowStepper, LengthRegulator, MelAnalyzer, Prompt, Resampler, Result,
    SpeakerEncoder, Vad, VelocityEstimator, Vocoder,
};

/// The flow model, either as a plain velocity field (solver updates on the host) or as a stepper that
/// applies the update itself (state resident on the device).
pub enum Estimator {
    Velocity(Box<dyn VelocityEstimator>),
    Fused(Box<dyn FlowStepper>),
}

impl Estimator {
    pub fn n_mels(&self) -> usize {
        match self {
            Estimator::Velocity(e) => e.n_mels(),
            Estimator::Fused(s) => s.n_mels(),
        }
    }
}
use mimic_dsp::Sola;
use ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView3, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

/// Euler solver over `t in [0, 1]` with classifier-free guidance, reproducing seed-vc's `solve_euler`.
#[derive(Debug, Clone, Copy)]
pub struct EulerCfg {
    pub steps: usize,
    /// guidance strength; `0.0` disables guidance (the unconditional half of the batch is still computed
    /// when the estimator is a fixed batch-2 graph, and simply ignored)
    pub cfg: f32,
}

impl EulerCfg {
    /// `mu`: `(1, L, D)` condition (prompt frames first); `prompt`: reference mel/style; `z`: `(1, n_mels, L)` noise.
    /// Returns the full `(1, n_mels, L)` trajectory end point; the first `prompt.frames()` frames are zero.
    pub fn solve(
        &self,
        est: &mut dyn VelocityEstimator,
        mu: ArrayView3<'_, f32>,
        prompt: &Prompt,
        z: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let p = prompt.frames();
        let (b, n_mels, l) = z.dim();
        if b != 1 || mu.shape()[0] != 1 {
            return Err(Error::Shape(
                "solver works on one utterance at a time".into(),
            ));
        }
        let mut x = z.clone();
        x.slice_mut(s![.., .., ..p]).fill(0.0);
        let mut prompt_x = Array3::<f32>::zeros((1, n_mels, l));
        prompt_x
            .slice_mut(s![.., .., ..p])
            .assign(&prompt.mel.slice(s![.., .., ..p]));
        let zeros3 = Array3::<f32>::zeros((1, n_mels, l));
        let zeros_mu = Array3::<f32>::zeros((1, l, mu.shape()[2]));
        let zeros_style = Array2::<f32>::zeros((1, prompt.style.shape()[1]));
        // batch 2: [conditional, unconditional]
        let prompt_x2 = concatenate(Axis(0), &[prompt_x.view(), zeros3.view()]).unwrap();
        let style2 = concatenate(Axis(0), &[prompt.style.view(), zeros_style.view()]).unwrap();
        let mu2 = concatenate(Axis(0), &[mu, zeros_mu.view()]).unwrap();
        let n = self.steps;
        est.set_context(prompt_x2.view(), style2.view(), mu2.view())?;
        let mut x2 = Array3::<f32>::zeros((2, n_mels, l));
        for k in 1..=n {
            let t = (k - 1) as f32 / n as f32;
            let dt = k as f32 / n as f32 - t;
            x2.slice_mut(s![0..1, .., ..]).assign(&x);
            x2.slice_mut(s![1..2, .., ..]).assign(&x);
            let t2 = Array1::from_vec(vec![t, t]);
            let v = est.velocity_step(x2.view(), t2.view())?;
            let v_c = v.slice(s![0..1, .., ..]);
            let v_u = v.slice(s![1..2, .., ..]);
            let dphi = if self.cfg > 0.0 {
                &v_c * (1.0 + self.cfg) - &v_u * self.cfg
            } else {
                v_c.to_owned()
            };
            x = x + dphi * dt;
            x.slice_mut(s![.., .., ..p]).fill(0.0);
        }
        Ok(x)
    }

    /// Same trajectory with a [`FlowStepper`] (update applied on the model side, state kept there).
    pub fn solve_fused(
        &self,
        st: &mut dyn FlowStepper,
        mu: ArrayView3<'_, f32>,
        prompt: &Prompt,
        z: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let p = prompt.frames();
        let (b, n_mels, l) = z.dim();
        if b != 1 || mu.shape()[0] != 1 {
            return Err(Error::Shape(
                "solver works on one utterance at a time".into(),
            ));
        }
        let mut x = z.clone();
        x.slice_mut(s![.., .., ..p]).fill(0.0);
        let mut prompt_x = Array3::<f32>::zeros((1, n_mels, l));
        prompt_x
            .slice_mut(s![.., .., ..p])
            .assign(&prompt.mel.slice(s![.., .., ..p]));
        let zeros3 = Array3::<f32>::zeros((1, n_mels, l));
        let zeros_mu = Array3::<f32>::zeros((1, l, mu.shape()[2]));
        let zeros_style = Array2::<f32>::zeros((1, prompt.style.shape()[1]));
        let x2 = concatenate(Axis(0), &[x.view(), x.view()]).unwrap();
        let prompt_x2 = concatenate(Axis(0), &[prompt_x.view(), zeros3.view()]).unwrap();
        let style2 = concatenate(Axis(0), &[prompt.style.view(), zeros_style.view()]).unwrap();
        let mu2 = concatenate(Axis(0), &[mu, zeros_mu.view()]).unwrap();
        st.begin(x2.view(), prompt_x2.view(), style2.view(), mu2.view())?;
        let n = self.steps;
        for k in 1..=n {
            let t = (k - 1) as f32 / n as f32;
            let dt = k as f32 / n as f32 - t;
            st.step(t, dt, self.cfg)?;
        }
        let out = st.state()?;
        Ok(out.slice(s![0..1, .., ..]).to_owned())
    }

    pub fn solve_any(
        &self,
        est: &mut Estimator,
        mu: ArrayView3<'_, f32>,
        prompt: &Prompt,
        z: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        match est {
            Estimator::Velocity(e) => self.solve(e.as_mut(), mu, prompt, z),
            Estimator::Fused(s) => self.solve_fused(s.as_mut(), mu, prompt, z),
        }
    }
}

/// The stages a seed-vc-style conversion needs. The `_ref` encoders handle the (fixed) reference length.
pub struct Stages {
    pub content: Box<dyn ContentEncoder>,
    pub content_ref: Box<dyn ContentEncoder>,
    pub regulator: Box<dyn LengthRegulator>,
    pub regulator_ref: Box<dyn LengthRegulator>,
    pub speaker: Box<dyn SpeakerEncoder>,
    pub estimator: Estimator,
    pub vocoder: Box<dyn Vocoder>,
    pub mel: Box<dyn MelAnalyzer>,
    /// acoustic rate -> content-encoder rate (22.05 k -> 16 k)
    pub to_content_rate: Box<dyn Resampler>,
}

/// Window geometry of the streaming algorithm, in acoustic-rate samples unless stated.
#[derive(Debug, Clone, Copy)]
pub struct StreamGeometry {
    pub acoustic_sr: usize,
    pub content_sr: usize,
    pub block_frame: usize,
    pub sola_buffer_frame: usize,
    pub sola_search_frame: usize,
    pub extra_frame_right: usize,
    pub window_16k: usize,
    pub block_16k: usize,
    pub skip_tail: usize,
    pub return_length: usize,
    pub ce_dit_frame_difference: usize,
    pub t_cond: usize,
    pub prompt_samples: usize,
    pub prompt_frames: usize,
}

impl StreamGeometry {
    pub fn block_seconds(&self) -> f32 {
        self.block_frame as f32 / self.acoustic_sr as f32
    }

    /// Algorithmic delay in acoustic-rate samples between the newest input sample and the emitted block
    /// (right context + cross-fade + SOLA search; SOLA may pull it back by up to `sola_search_frame`).
    pub fn delay_samples(&self) -> usize {
        self.extra_frame_right + self.sola_buffer_frame + self.sola_search_frame
    }
}

impl Stages {
    /// Compute the reference prompt from a reference waveform at the acoustic sample rate.
    /// The reference is truncated (or zero-padded) to `geo.prompt_samples`.
    pub fn prepare_prompt(&mut self, reference: &[f32], geo: &StreamGeometry) -> Result<Prompt> {
        let mut r = reference.to_vec();
        r.resize(geo.prompt_samples, 0.0);
        let r16 = self.to_content_rate.resample(&r);
        let r16_arr = Array2::from_shape_vec((1, r16.len()), r16.clone()).unwrap();
        let s_ori = self.content_ref.encode(r16_arr.view())?;
        let style = self.speaker.embed(&r16)?;
        let mel = self.mel.analyze(&r);
        let (nm, pf) = mel.dim();
        if pf != geo.prompt_frames {
            return Err(Error::Shape(format!(
                "prompt mel has {pf} frames, geometry says {}",
                geo.prompt_frames
            )));
        }
        let cond = self.regulator_ref.regulate(s_ori.view(), pf)?;
        let e = self.speaker.embedding_dim();
        Ok(Prompt {
            cond,
            mel: mel.into_shape_with_order((1, nm, pf)).unwrap(),
            style: style.into_shape_with_order((1, e)).unwrap(),
        })
    }
}

/// Per-block timing of the last `push`, milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockTiming {
    pub content_ms: f32,
    pub regulate_ms: f32,
    pub solve_ms: f32,
    pub vocode_ms: f32,
    pub total_ms: f32,
}

/// Streaming state for one voice/call. Holds no model; advanced by a [`Runner`].
pub struct Session {
    geo: StreamGeometry,
    prompt: Prompt,
    pub solver: EulerCfg,
    /// rolling 16 kHz analysis window
    window16: Vec<f32>,
    sola: Sola,
    rng: StdRng,
    vad: Option<Box<dyn Vad>>,
    /// VAD decisions in flight: the block emitted now carries input from `delay_samples` ago, so the gate
    /// that applies to it is the decision made that many blocks earlier (9 blocks at a 2 s right context,
    /// 0 at 0.02 s). Gating with the current decision chops speech wherever the pause pattern of "now"
    /// differs from the pause pattern of the audio being played.
    gate_delay: std::collections::VecDeque<f32>,
    gate_prev: f32,
    pub last_timing: BlockTiming,
    /// Blocks converted by [`Runner::push`].
    pub blocks: usize,
    /// Blocks advanced by [`Session::roll`] (no model work).
    pub rolled: usize,
}

impl Session {
    pub fn new(geo: StreamGeometry, prompt: Prompt, solver: EulerCfg, seed: u64) -> Self {
        let sola = Sola::new(
            geo.block_frame,
            geo.sola_buffer_frame,
            geo.sola_search_frame,
        );
        Self {
            geo,
            prompt,
            solver,
            window16: vec![0.0; geo.window_16k],
            sola,
            rng: StdRng::seed_from_u64(seed),
            vad: None,
            gate_delay: std::collections::VecDeque::new(),
            gate_prev: 1.0,
            last_timing: BlockTiming::default(),
            blocks: 0,
            rolled: 0,
        }
    }

    pub fn with_vad(mut self, vad: Box<dyn Vad>) -> Self {
        self.vad = Some(vad);
        // nearest whole block of algorithmic delay; the outputs that precede the first real decision are the
        // window's leading silence anyway, so they start gated
        let blocks = (self.geo.delay_samples() + self.geo.block_frame / 2) / self.geo.block_frame;
        self.gate_delay = std::iter::repeat_n(0.0f32, blocks).collect();
        self
    }

    /// Blocks between a VAD decision and the output it gates (see `gate_delay`).
    pub fn gate_delay_blocks(&self) -> usize {
        self.gate_delay.len()
    }

    pub fn geometry(&self) -> &StreamGeometry {
        &self.geo
    }

    /// Replace the reference prompt (voice change); the analysis window and SOLA state are kept.
    pub fn set_prompt(&mut self, prompt: Prompt) {
        self.prompt = prompt;
    }

    pub fn delay_samples(&self) -> usize {
        self.geo.delay_samples()
    }

    /// The rolling 16 kHz analysis window (oldest sample first).
    pub fn window16(&self) -> &[f32] {
        &self.window16
    }

    /// Advance the session through one quiet block **without running the model**: for a stream that is idle
    /// (a call where nobody speaks) this is what to call instead of [`Runner::push`], and it costs no GPU.
    /// Takes exactly `block_16k` samples like `push`, returns no audio (the caller plays silence).
    ///
    /// Everything `push` would have done to the state is done here, so a later `push` sees the same session
    /// as if `push` had been called throughout:
    /// - the analysis window is rolled and holds the real quiet audio (no stale pre-pause speech);
    /// - the VAD, if any, is fed the block so its hysteresis and context continue, and its real decision
    ///   enters the alignment queue (rule 7: decisions are aligned to the output they gate), so if the gate
    ///   opens during the rolled stretch the first converted block is not clipped;
    /// - the solver noise advances by one block's draw, so the first converted block after the pause is the
    ///   same noise sample a continuous run would have used;
    /// - the SOLA tail is cleared and the gate is marked closed, so the next `push` cross-fades in from
    ///   silence rather than from the tail of the last converted speech.
    ///
    /// Only the stitching of the first block after the pause differs from a continuous run (it fades in from
    /// zero instead of from converted silence). Does not touch `blocks` or `last_timing`; counts in `rolled`.
    pub fn roll(&mut self, block16: &[f32]) -> Result<()> {
        let g = self.geo;
        if block16.len() != g.block_16k {
            return Err(Error::Shape(format!(
                "roll expects {} samples, got {}",
                g.block_16k,
                block16.len()
            )));
        }
        if let Some(v) = self.vad.as_mut() {
            let now = v.process(block16);
            self.gate_delay.push_back(now);
            self.gate_delay.pop_front();
        }
        self.window16.copy_within(g.block_16k.., 0);
        let n = self.window16.len();
        self.window16[n - g.block_16k..].copy_from_slice(block16);
        // the same number of draws `push` makes for one block: n_mels x (prompt frames + t_cond)
        let draws = self.prompt.mel.shape()[1] * (self.prompt.cond.shape()[1] + g.t_cond);
        for _ in 0..draws {
            mimic_dsp::noise::standard_normal(&mut self.rng);
        }
        self.sola.reset();
        self.gate_prev = 0.0;
        self.rolled += 1;
        Ok(())
    }
}

/// Owns the model stages and advances sessions one block at a time (round-robin by the caller).
pub struct Runner {
    pub stages: Stages,
}

impl Runner {
    pub fn new(stages: Stages) -> Self {
        Self { stages }
    }

    /// Push exactly `block_16k` new 16 kHz samples into `sess`; returns `block_frame` converted samples at the
    /// acoustic rate. When the session has a VAD and it is closed, the block is still converted (the window must
    /// keep rolling) but the output is faded to silence. A stream the caller knows to be idle can be advanced
    /// with [`Session::roll`] instead, which rolls the same state without running the model.
    pub fn push(&mut self, sess: &mut Session, block16: &[f32]) -> Result<Vec<f32>> {
        let g = sess.geo;
        if block16.len() != g.block_16k {
            return Err(Error::Shape(format!(
                "push expects {} samples, got {}",
                g.block_16k,
                block16.len()
            )));
        }
        let t0 = Instant::now();
        let gate = match sess.vad.as_mut() {
            Some(v) => {
                let now = v.process(block16);
                sess.gate_delay.push_back(now);
                let cur = sess.gate_delay.pop_front().unwrap_or(now);
                // one block of pre-roll when the delay affords lookahead: an onset that the VAD only
                // confirms on the next block is not clipped
                let next = sess.gate_delay.front().copied().unwrap_or(cur);
                cur.max(next)
            }
            None => 1.0,
        };
        // roll the window
        sess.window16.copy_within(g.block_16k.., 0);
        let n = sess.window16.len();
        sess.window16[n - g.block_16k..].copy_from_slice(block16);

        let w = Array2::from_shape_vec((1, n), sess.window16.clone()).unwrap();
        let s_alt = self.stages.content.encode(w.view())?;
        let t1 = Instant::now();
        let s_cut = s_alt.slice(s![.., g.ce_dit_frame_difference.., ..]);
        let cond = self.stages.regulator.regulate(s_cut, g.t_cond)?;
        let mu = concatenate(Axis(1), &[sess.prompt.cond.view(), cond.view()]).unwrap();
        let t2 = Instant::now();
        let l = mu.shape()[1];
        let n_mels = self.stages.estimator.n_mels();
        let z = Array3::from_shape_fn((1, n_mels, l), |_| {
            mimic_dsp::noise::standard_normal(&mut sess.rng)
        });
        let x = sess
            .solver
            .solve_any(&mut self.stages.estimator, mu.view(), &sess.prompt, &z)?;
        let mel_out = x.slice(s![.., .., g.prompt_frames..]).to_owned();
        let t3 = Instant::now();
        let wav = self.stages.vocoder.vocode(mel_out.view())?; // (1, t_cond * hop)
        let t4 = Instant::now();
        let wav = wav.row(0);
        let output_len = g.return_length * g.acoustic_sr / 50;
        let tail_len = g.skip_tail * g.acoustic_sr / 50;
        let total = wav.len();
        if total < output_len + tail_len {
            return Err(Error::Shape(format!(
                "vocoder produced {total} samples, need {}",
                output_len + tail_len
            )));
        }
        let infer = wav.slice(s![total - output_len - tail_len..total - tail_len]);
        let mut out = sess.sola.process(infer.as_slice().unwrap());
        // gate: cross-fade-length ramp (~40 ms) at the start of the block on transitions, hard zero while
        // closed. A ramp over the whole block softened every onset by a quarter second.
        if gate != sess.gate_prev {
            let r = g.sola_buffer_frame.clamp(1, out.len().max(1)) as f32;
            for (i, v) in out.iter_mut().enumerate() {
                let t = (i as f32 / r).min(1.0);
                *v *= sess.gate_prev + (gate - sess.gate_prev) * t;
            }
        } else if gate == 0.0 {
            out.iter_mut().for_each(|v| *v = 0.0);
        }
        sess.gate_prev = gate;
        sess.blocks += 1;
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1e3;
        sess.last_timing = BlockTiming {
            content_ms: ms(t0, t1),
            regulate_ms: ms(t1, t2),
            solve_ms: ms(t2, t3),
            vocode_ms: ms(t3, t4),
            total_ms: ms(t0, t4),
        };
        Ok(out)
    }
}

/// Whole-file conversion through the streaming code path.
pub struct Converter;

impl Converter {
    /// `source16k`: source speech at the content-encoder rate (16 kHz). Returns audio at the acoustic rate,
    /// aligned to the source (the streaming delay is removed) and trimmed to the source duration.
    pub fn convert(
        runner: &mut Runner,
        sess: &mut Session,
        source16k: &[f32],
        mut on_block: impl FnMut(usize, &BlockTiming),
    ) -> Result<Vec<f32>> {
        let geo = sess.geo;
        let delay = sess.delay_samples();
        let n_src_blocks = source16k.len().div_ceil(geo.block_16k);
        let n_flush = delay.div_ceil(geo.block_frame) + 1;
        let mut out = Vec::with_capacity((n_src_blocks + n_flush) * geo.block_frame);
        let mut buf = vec![0.0f32; geo.block_16k];
        for i in 0..n_src_blocks + n_flush {
            buf.iter_mut().for_each(|v| *v = 0.0);
            if i < n_src_blocks {
                let st = i * geo.block_16k;
                let en = (st + geo.block_16k).min(source16k.len());
                buf[..en - st].copy_from_slice(&source16k[st..en]);
            }
            let y = runner.push(sess, &buf)?;
            on_block(i, &sess.last_timing);
            out.extend_from_slice(&y);
        }
        let target_len =
            (source16k.len() as u64 * geo.acoustic_sr as u64 / geo.content_sr as u64) as usize;
        let start = delay.min(out.len());
        let mut aligned: Vec<f32> = out[start..].to_vec();
        aligned.truncate(target_len);
        Ok(aligned)
    }
}
