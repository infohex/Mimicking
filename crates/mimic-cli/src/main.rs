//! `mimic` command line: file-to-file conversion, real-time simulation, model verification.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use mimic_dsp::EnergyVad;
use mimic_onnx::{Device, OnnxRuntime};
use mimic_pipeline::{BlockTiming, Converter, EulerCfg, Runner, Session, StreamGeometry};
use mimic_model::{ModelFiles, ModelSet, CONTENT_SR, SR};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "mimic",
    version,
    about = "Mimicking: Rust zero-shot voice conversion"
)]
struct Cli {
    /// Path to onnxruntime.dll / libonnxruntime.so (default: MIMIC_ORT_DIR, the exe folder/runtime/onnxruntime, ORT_DYLIB_PATH)
    #[arg(long, global = true)]
    ort: Option<PathBuf>,
    /// Model set directory (default: the first set with a manifest under <exe folder>/models, then ./models; *-lowlat preferred)
    #[arg(long, global = true)]
    models: Option<PathBuf>,
    /// CUDA device ordinal; negative = CPU
    #[arg(long, global = true, default_value_t = 0)]
    gpu: i32,
    /// Disable CUDA Graph capture/replay (device-resident I/O binding stays on)
    #[arg(long, global = true, default_value_t = false)]
    no_cuda_graph: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    #[arg(long, default_value_t = 10)]
    steps: usize,
    #[arg(long, default_value_t = 0.7)]
    cfg: f32,
    /// Block length in seconds (default: what the model directory's manifest was exported for).
    #[arg(long)]
    block: Option<f32>,
    /// Right context in seconds (default: from the manifest; the graphs are static, so this must match the export).
    #[arg(long)]
    extra_right: Option<f32>,
    #[arg(long, default_value_t = 1234)]
    seed: u64,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a source file to the voice of a reference file (offline, through the streaming code path).
    Convert {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Simulate N concurrent real-time sessions: each --source file is one call, fed block by block.
    Stream {
        /// one reference per session (reused cyclically if fewer than sources)
        #[arg(long, required = true, num_args = 1..)]
        reference: Vec<PathBuf>,
        #[arg(long, required = true, num_args = 1..)]
        source: Vec<PathBuf>,
        /// output directory for per-session wavs (omit to discard output)
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// pace input at real time (sleep until each block period) instead of running flat out
        #[arg(long, default_value_t = false)]
        realtime: bool,
        /// energy VAD gate on each session
        #[arg(long, default_value_t = false)]
        vad: bool,
        /// stop after this many seconds of source audio per session
        #[arg(long)]
        seconds: Option<f32>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Inspect or verify the model directory against manifest.json.
    Models {
        #[command(subcommand)]
        what: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List the manifest entries and the configuration the graphs were exported for.
    List,
    /// Check presence, size and sha256 of every file in the manifest.
    Verify,
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.unwrap() as f32 * scale)
                .collect()
        }
    };
    let mono: Vec<f32> = if ch == 1 {
        samples
    } else {
        samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn write_wav(path: &Path, sr: u32, x: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn to_rate(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    mimic_dsp::SincResampler::new(from as usize, to as usize).resample(x)
}

/// `--models` default: the first directory carrying a `manifest.json` under `<exe dir>/models`, then `./models`
/// (the root itself, else its subdirectories; a `*-lowlat` set wins, otherwise the first by name), so a packaged
/// runtime with one set and a checkout with several both work. Falls back to `models` so the error names the place.
fn default_models_dir() -> PathBuf {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("models"));
        }
    }
    roots.push(PathBuf::from("models"));
    for root in &roots {
        if root.join("manifest.json").is_file() {
            return root.clone();
        }
        if let Ok(rd) = std::fs::read_dir(root) {
            let mut sets: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.join("manifest.json").is_file()).collect();
            sets.sort();
            if let Some(p) = sets.iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().ends_with("-lowlat"))) {
                return p.clone();
            }
            if let Some(p) = sets.into_iter().next() {
                return p;
            }
        }
    }
    PathBuf::from("models")
}

fn load_runner(
    rt: &OnnxRuntime,
    files: &ModelFiles,
    m: &ModelArgs,
) -> Result<(Runner, StreamGeometry)> {
    let mut config = files.config();
    if let Some(b) = m.block {
        config.block_time = b;
    }
    if let Some(r) = m.extra_right {
        config.extra_time_right = r;
    }
    let t0 = Instant::now();
    let model = ModelSet::load(rt, files, config)?;
    eprintln!(
        "models loaded in {:.1}s; block {:.3}s, window {:.2}s, prompt {} frames, seq {}; flow: {}",
        t0.elapsed().as_secs_f32(),
        model.shapes.block_frame as f32 / SR as f32,
        model.shapes.window as f32 / SR as f32,
        model.shapes.prompt_frames,
        model.shapes.seq_len,
        if model.stepper.is_some() {
            "fused Euler step on device"
        } else {
            "velocity graph + host Euler"
        }
    );
    let (stages, geo) = model.into_stages();
    Ok((Runner::new(stages), geo))
}

fn prompt_from_file(
    runner: &mut Runner,
    geo: &StreamGeometry,
    path: &Path,
) -> Result<mimic_core::Prompt> {
    let (r, rsr) = read_wav_mono(path)?;
    let r22 = to_rate(&r, rsr, SR as u32);
    if r22.len() < geo.prompt_samples {
        bail!(
            "reference {} shorter than {} s",
            path.display(),
            geo.prompt_samples as f32 / SR as f32
        );
    }
    let t = Instant::now();
    let p = runner.stages.prepare_prompt(&r22, geo)?;
    eprintln!(
        "prompt {} prepared in {:.0} ms",
        path.display(),
        t.elapsed().as_secs_f32() * 1e3
    );
    Ok(p)
}

#[derive(Default)]
struct Stats {
    totals: Vec<f32>,
    sum: BlockTiming,
}

impl Stats {
    fn add(&mut self, t: &BlockTiming) {
        self.totals.push(t.total_ms);
        self.sum.content_ms += t.content_ms;
        self.sum.regulate_ms += t.regulate_ms;
        self.sum.solve_ms += t.solve_ms;
        self.sum.vocode_ms += t.vocode_ms;
        self.sum.total_ms += t.total_ms;
    }
    fn report(&self, label: &str) {
        if self.totals.is_empty() {
            return;
        }
        let mut s = self.totals.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        let pct = |p: f32| s[((n as f32 - 1.0) * p).round() as usize];
        let nf = n as f32;
        eprintln!(
            "{label}: {n} blocks; per block p50 {:.1} p95 {:.1} max {:.1} ms; mean content {:.1} regulate {:.1} solve {:.1} vocode {:.1} total {:.1} ms",
            pct(0.5),
            pct(0.95),
            s[n - 1],
            self.sum.content_ms / nf,
            self.sum.regulate_ms / nf,
            self.sum.solve_ms / nf,
            self.sum.vocode_ms / nf,
            self.sum.total_ms / nf
        );
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let files = ModelFiles::new(cli.models.clone().unwrap_or_else(default_models_dir));

    if let Cmd::Models { what } = &cli.cmd {
        match what {
            ModelsCmd::List => {
                let m = files.read_manifest()?;
                println!("{} {}", m.name, m.version);
                println!("config: {}", serde_json_compact(&m.config));
                for e in &m.files {
                    println!(
                        "  {:28} {:>11} bytes  {}  {}",
                        e.file,
                        e.bytes,
                        &e.sha256[..12],
                        e.role
                    );
                }
                println!("provenance: {}", serde_json_compact(&m.provenance));
            }
            ModelsCmd::Verify => {
                let t = Instant::now();
                let entries = files.verify()?;
                println!(
                    "ok: {} files in {} verified in {:.1}s",
                    entries.len(),
                    files.dir.display(),
                    t.elapsed().as_secs_f32()
                );
            }
        }
        return Ok(());
    }

    let device = if cli.gpu < 0 {
        Device::Cpu
    } else {
        Device::Cuda(cli.gpu)
    };
    let rt = OnnxRuntime::discover(cli.ort.as_deref(), device)?.with_cuda_graph(!cli.no_cuda_graph);
    match cli.cmd {
        Cmd::Models { .. } => unreachable!(),
        Cmd::Convert {
            reference,
            source,
            out,
            model,
        } => {
            let (mut runner, geo) = load_runner(&rt, &files, &model)?;
            let prompt = prompt_from_file(&mut runner, &geo, &reference)?;
            let (s, ssr) = read_wav_mono(&source)?;
            let s16 = to_rate(&s, ssr, CONTENT_SR as u32);
            let mut sess = Session::new(
                geo,
                prompt,
                EulerCfg {
                    steps: model.steps,
                    cfg: model.cfg,
                },
                model.seed,
            );
            let mut stats = Stats::default();
            let t2 = Instant::now();
            let y = Converter::convert(&mut runner, &mut sess, &s16, |_, t| stats.add(t))?;
            let el = t2.elapsed().as_secs_f32();
            stats.report("convert");
            eprintln!("RTF {:.3}", el / (s16.len() as f32 / CONTENT_SR as f32));
            write_wav(&out, SR as u32, &y)?;
            eprintln!(
                "wrote {} ({:.2}s)",
                out.display(),
                y.len() as f32 / SR as f32
            );
        }
        Cmd::Stream {
            reference,
            source,
            out_dir,
            realtime,
            vad,
            seconds,
            model,
        } => {
            let (mut runner, geo) = load_runner(&rt, &files, &model)?;
            let mut prompts = Vec::new();
            for r in &reference {
                prompts.push(prompt_from_file(&mut runner, &geo, r)?);
            }
            let mut sessions = Vec::new();
            let mut inputs: Vec<Vec<f32>> = Vec::new();
            for (i, src) in source.iter().enumerate() {
                let (s, ssr) = read_wav_mono(src)?;
                let mut s16 = to_rate(&s, ssr, CONTENT_SR as u32);
                if let Some(sec) = seconds {
                    s16.truncate((sec * CONTENT_SR as f32) as usize);
                }
                inputs.push(s16);
                let mut sess = Session::new(
                    geo,
                    prompts[i % prompts.len()].clone(),
                    EulerCfg {
                        steps: model.steps,
                        cfg: model.cfg,
                    },
                    model.seed + i as u64,
                );
                if vad {
                    sess = sess.with_vad(Box::new(EnergyVad::default()));
                }
                sessions.push(sess);
            }
            let n_sess = sessions.len();
            let n_blocks = inputs
                .iter()
                .map(|s| s.len().div_ceil(geo.block_16k))
                .max()
                .unwrap_or(0);
            let period = Duration::from_secs_f32(geo.block_seconds());
            eprintln!(
                "{n_sess} session(s), {n_blocks} blocks each, block period {:.0} ms, realtime={realtime}, vad={vad}",
                period.as_secs_f32() * 1e3
            );
            let mut outs: Vec<Vec<f32>> = vec![Vec::new(); n_sess];
            let mut stats: Vec<Stats> = (0..n_sess).map(|_| Stats::default()).collect();
            let mut round_ms: Vec<f32> = Vec::with_capacity(n_blocks);
            let mut misses = 0usize;
            let mut buf = vec![0.0f32; geo.block_16k];
            let start = Instant::now();
            for b in 0..n_blocks {
                let deadline = start + period * (b as u32 + 1);
                let t_round = Instant::now();
                for (i, sess) in sessions.iter_mut().enumerate() {
                    buf.iter_mut().for_each(|v| *v = 0.0);
                    let st = b * geo.block_16k;
                    if st < inputs[i].len() {
                        let en = (st + geo.block_16k).min(inputs[i].len());
                        buf[..en - st].copy_from_slice(&inputs[i][st..en]);
                    }
                    let y = runner.push(sess, &buf)?;
                    stats[i].add(&sess.last_timing);
                    if out_dir.is_some() {
                        outs[i].extend_from_slice(&y);
                    }
                }
                let r = t_round.elapsed().as_secs_f32() * 1e3;
                round_ms.push(r);
                if realtime {
                    let now = Instant::now();
                    if now > deadline {
                        misses += 1;
                    } else {
                        std::thread::sleep(deadline - now);
                    }
                } else if r > period.as_secs_f32() * 1e3 {
                    misses += 1;
                }
            }
            let wall = start.elapsed().as_secs_f32();
            for (i, s) in stats.iter().enumerate() {
                s.report(&format!("session {i}"));
            }
            let mut rs = round_ms.clone();
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = rs.len().max(1);
            let pct = |p: f32| rs[((n as f32 - 1.0) * p).round() as usize];
            eprintln!(
                "round (all {n_sess} sessions): p50 {:.1} p95 {:.1} max {:.1} ms vs period {:.0} ms; deadline misses {misses}/{n_blocks}; wall {wall:.1}s; aggregate RTF {:.3}",
                pct(0.5),
                pct(0.95),
                rs[n - 1],
                period.as_secs_f32() * 1e3,
                wall / (n_blocks as f32 * geo.block_seconds())
            );
            if let Some(dir) = out_dir {
                std::fs::create_dir_all(&dir)?;
                for (i, y) in outs.iter().enumerate() {
                    let p = dir.join(format!("session{i}.wav"));
                    write_wav(&p, SR as u32, &y[geo.delay_samples().min(y.len())..])?;
                }
                eprintln!("wrote {} session file(s) to {}", n_sess, dir.display());
            }
        }
    }
    Ok(())
}

fn serde_json_compact(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}
