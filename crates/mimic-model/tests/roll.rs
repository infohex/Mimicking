//! `Session::roll` against the real graphs: a stream that is pushed, rolled through a quiet stretch and pushed
//! again must sound like one that was pushed throughout, and must not carry pre-pause speech. Needs a GPU and
//! an onnxruntime dylib, like `parity.rs`:
//!   MIMIC_ORT_DYLIB=<onnxruntime.dll>  MIMIC_GPU=0  cargo test -p mimic-model --release --test roll -- --nocapture
//! Skipped (not failed) when the dylib or the graphs are missing. `MIMIC_MODELS` picks another set.
//!
//! Two runs of the same session never match sample for sample: the vocoder draws its overtone phases and its
//! noise from the runner's own generator, and SOLA then picks different lags. The comparison is therefore a
//! spectral-envelope distance (phase- and lag-invariant), and the yardstick is the spread that identical
//! sessions show on the blocks before the pause.

use mimic_dsp::EnergyVad;
use mimic_model::{Device, ModelFiles, ModelSet, OnnxRuntime, StreamGeometry};
use mimic_pipeline::{EulerCfg, Runner, Session};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn models_dir() -> PathBuf {
    std::env::var_os("MIMIC_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("models/Mimicking-v2.0-lowlat"))
}

struct Bench {
    runner: Runner,
    geo: StreamGeometry,
    prompt: mimic_core::Prompt,
}

/// A harmonic "voice": 120 Hz with vibrato, eight harmonics, a slow envelope that never drops below half.
fn synthetic_voice(sr: usize, seconds: f32) -> Vec<f32> {
    let n = (sr as f32 * seconds) as usize;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f0 = 120.0 + 6.0 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
            phase += 2.0 * std::f32::consts::PI * f0 / sr as f32;
            let env = 0.75 + 0.25 * (2.0 * std::f32::consts::PI * 2.3 * t).sin();
            let mut v = 0.0;
            for h in 1..=8 {
                v += (phase * h as f32).sin() / h as f32;
            }
            0.25 * env * v
        })
        .collect()
}

/// Built once per process (one ONNX Runtime environment, rule 1), so every check runs from one test.
fn bench() -> Option<Bench> {
    let Some(dylib) = std::env::var_os("MIMIC_ORT_DYLIB") else {
        eprintln!("MIMIC_ORT_DYLIB not set, skipping roll tests");
        return None;
    };
    let gpu: i32 = std::env::var("MIMIC_GPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let files = ModelFiles::new(models_dir());
    if !files.dit_step().exists() {
        eprintln!("graphs missing in {}, skipping", files.dir.display());
        return None;
    }
    let rt = OnnxRuntime::load(dylib, Device::Cuda(gpu)).expect("load onnxruntime");
    let config = files.config();
    let m = ModelSet::load(&rt, &files, config).expect("load models");
    let (stages, geo) = m.into_stages();
    let mut runner = Runner::new(stages);
    let reference = synthetic_voice(geo.acoustic_sr, 3.2);
    let prompt = runner.stages.prepare_prompt(&reference, &geo).expect("prompt");
    eprintln!(
        "set {} loaded: block {} / 16k {}, window 16k {}, delay {} samples",
        files.dir.display(),
        geo.block_frame,
        geo.block_16k,
        geo.window_16k,
        geo.delay_samples()
    );
    Some(Bench { runner, geo, prompt })
}

fn session(b: &Bench, seed: u64, vad: bool) -> Session {
    let s = Session::new(b.geo, b.prompt.clone(), EulerCfg { steps: 10, cfg: 0.7 }, seed);
    if vad {
        s.with_vad(Box::new(EnergyVad::default()))
    } else {
        s
    }
}

/// Source blocks at 16 kHz: `lead` blocks of near-silence (about -80 dBFS noise, so a gate's floor settles),
/// `pre` blocks of voice, `quiet` blocks of near-silence, `post` blocks of voice.
fn source_blocks(geo: &StreamGeometry, lead: usize, pre: usize, quiet: usize, post: usize) -> Vec<Vec<f32>> {
    let voice = synthetic_voice(
        geo.content_sr,
        (pre + post + 1) as f32 * geo.block_16k as f32 / geo.content_sr as f32,
    );
    let mut blocks = Vec::new();
    let mut seed = 12345u32;
    let mut noise = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2e-4
    };
    for _ in 0..lead {
        blocks.push((0..geo.block_16k).map(|_| noise()).collect());
    }
    for i in 0..pre {
        blocks.push(voice[i * geo.block_16k..(i + 1) * geo.block_16k].to_vec());
    }
    for _ in 0..quiet {
        blocks.push((0..geo.block_16k).map(|_| noise()).collect());
    }
    for i in pre..pre + post {
        blocks.push(voice[i * geo.block_16k..(i + 1) * geo.block_16k].to_vec());
    }
    blocks
}

fn rms_db(x: &[f32]) -> f32 {
    let e = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    10.0 * (e + 1e-20).log10()
}

/// Band energies in dB of one block: 32 log-spaced bands between 100 Hz and 8 kHz from a plain DFT.
fn band_db(x: &[f32], sr: usize) -> Vec<f32> {
    const BANDS: usize = 32;
    let n = x.len();
    let edges: Vec<f32> = (0..=BANDS)
        .map(|i| 100.0 * (80.0f32).powf(i as f32 / BANDS as f32))
        .collect();
    let mut bands = vec![0.0f32; BANDS];
    let k_max = (edges[BANDS] * n as f32 / sr as f32) as usize;
    let k_min = (edges[0] * n as f32 / sr as f32) as usize;
    for k in k_min..=k_max {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        let w = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        for (i, &v) in x.iter().enumerate() {
            let a = w * i as f32;
            re += v * a.cos();
            im += v * a.sin();
        }
        let f = k as f32 * sr as f32 / n as f32;
        if let Some(b) = edges.windows(2).position(|e| f >= e[0] && f < e[1]) {
            bands[b] += re * re + im * im;
        }
    }
    let peak = bands.iter().cloned().fold(0.0f32, f32::max);
    bands.iter().map(|e| 10.0 * (e.max(peak * 1e-4) + 1e-12).log10()).collect()
}

/// RMS over bands of the dB difference between two blocks' envelopes.
fn spectral_distance(a: &[f32], b: &[f32], sr: usize) -> f32 {
    let (da, db) = (band_db(a, sr), band_db(b, sr));
    let s: f32 = da.iter().zip(&db).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / da.len() as f32).sqrt()
}

/// Continuous pushes vs push / roll / push, for a short pause and for a pause longer than the window, with and
/// without a gate. The yardstick is a second continuous session with the same seed (`same`): after the pause
/// the rolled session must be as close to the continuous one as that repeat is, pooled over the four cases
/// because single blocks swing by several dB from run to run (the graphs are not bit-reproducible). The
/// pre-pause blocks, where all sessions are identical, give the null spread for reference, and a run with
/// another noise seed is printed as the yardstick for "different".
fn rolled_stream_matches_continuous_pushes(b: &mut Bench) {
    let geo = b.geo;
    let sr = geo.acoustic_sr;
    let window_blocks = geo.window_16k.div_ceil(geo.block_16k);
    let lead = 4;
    let pre = window_blocks + 4;
    let post = 4;
    let (mut pooled_roll, mut pooled_same) = (Vec::new(), Vec::new());
    for &quiet in &[2usize, window_blocks + 2] {
        for &vad in &[false, true] {
            let blocks = source_blocks(&geo, lead, pre, quiet, post);
            let mut cont = session(b, 7, vad);
            let mut same = session(b, 7, vad);
            let mut other = session(b, 8, vad);
            let mut rolled = session(b, 7, vad);
            let (mut y_cont, mut y_same, mut y_other, mut y_roll) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, blk) in blocks.iter().enumerate() {
                y_cont.push(b.runner.push(&mut cont, blk).unwrap());
                y_same.push(b.runner.push(&mut same, blk).unwrap());
                y_other.push(b.runner.push(&mut other, blk).unwrap());
                if i >= lead + pre && i < lead + pre + quiet {
                    rolled.roll(blk).unwrap();
                    y_roll.push(Vec::new());
                } else {
                    y_roll.push(b.runner.push(&mut rolled, blk).unwrap());
                }
            }
            assert_eq!(rolled.rolled, quiet);
            assert_eq!(rolled.blocks, lead + pre + post);
            assert_eq!(rolled.window16(), cont.window16(), "windows differ (quiet {quiet}, vad {vad})");
            // null spread: blocks before the pause where both sessions are audible (the gate may hold the
            // first ones closed)
            let audible = |y: &[f32]| rms_db(y) > -60.0;
            let mut null = Vec::new();
            for k in lead + 2..lead + pre {
                if audible(&y_cont[k]) {
                    assert!(audible(&y_roll[k]), "pre-pause block {k}: gate differs");
                    null.push(spectral_distance(&y_cont[k], &y_roll[k], sr));
                }
            }
            let null_max = null.iter().cloned().fold(0.0f32, f32::max);
            let null_mean = null.iter().sum::<f32>() / null.len().max(1) as f32;
            assert!(null.len() >= 4, "too few audible pre-pause blocks ({})", null.len());
            // the output block that carries the first post-pause input (gate delay included)
            let first = lead + pre + quiet + cont.gate_delay_blocks();
            for k in first..blocks.len() {
                assert_eq!(audible(&y_cont[k]), audible(&y_roll[k]), "block {k}: gate differs after the pause");
                if !audible(&y_cont[k]) {
                    continue;
                }
                let d_roll = spectral_distance(&y_cont[k], &y_roll[k], sr);
                let d_same = spectral_distance(&y_cont[k], &y_same[k], sr);
                let d_other = spectral_distance(&y_cont[k], &y_other[k], sr);
                pooled_roll.push(d_roll);
                pooled_same.push(d_same);
                eprintln!(
                    "quiet {quiet:>2} vad {vad:<5} block {k}{}: {:.1} dBFS, envelope distance rolled {d_roll:.2} dB, same seed {d_same:.2} dB, other seed {d_other:.2} dB (pre-pause null: mean {null_mean:.2}, max {null_max:.2} dB)",
                    if k == first { " (first after pause)" } else { "" },
                    rms_db(&y_cont[k]),
                );
            }
            // The first block after the pause: its body must be at the level of the continuous run, and its
            // seam (the cross-fade length at the start) must not be louder than the continuous one, which
            // cross-fades from the tail of the converted silence while the rolled one fades in from zero.
            let h = geo.sola_buffer_frame;
            let (c_head, c_body) = (rms_db(&y_cont[first][..h]), rms_db(&y_cont[first][h..]));
            let (r_head, r_body) = (rms_db(&y_roll[first][..h]), rms_db(&y_roll[first][h..]));
            eprintln!(
                "quiet {quiet:>2} vad {vad:<5} first block seam / body: continuous {c_head:.1} / {c_body:.1}, rolled {r_head:.1} / {r_body:.1} dBFS"
            );
            assert!(
                (c_body - r_body).abs() <= 1.5,
                "first block after the pause: rolled body {r_body:.1} dBFS vs continuous {c_body:.1} dBFS (quiet {quiet}, vad {vad})"
            );
            assert!(
                r_head <= c_head.max(-60.0) + 1.0,
                "first block after the pause: rolled seam {r_head:.1} dBFS louder than continuous {c_head:.1} dBFS (quiet {quiet}, vad {vad})"
            );
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (m_roll, m_same) = (mean(&pooled_roll), mean(&pooled_same));
    eprintln!(
        "after the pause, over {} blocks: rolled {m_roll:.2} dB from continuous, same-seed repeat {m_same:.2} dB",
        pooled_roll.len()
    );
    assert!(pooled_roll.len() >= 8, "too few audible blocks after the pauses ({})", pooled_roll.len());
    assert!(
        m_roll <= m_same + 1.5,
        "rolled sessions drift from the continuous ones: {m_roll:.2} dB vs a same-seed repeat {m_same:.2} dB"
    );
}

/// Voice, then a whole window of near-silence, then more near-silence (no gate): the session that rolled
/// through the pause must come out as quiet as the one that pushed through it. Whatever the model makes of a
/// silent window is the same on both sides; what must not happen is the rolled side carrying the voice.
fn rolled_pause_leaves_no_stale_speech(b: &mut Bench) {
    let geo = b.geo;
    let window_blocks = geo.window_16k.div_ceil(geo.block_16k);
    let pre = window_blocks + 2;
    let quiet = window_blocks + 2;
    let tail = 3;
    let blocks = source_blocks(&geo, 0, pre, quiet + tail, 0);
    let mut cont = session(b, 7, false);
    let mut rolled = session(b, 7, false);
    let mut voiced = Vec::new();
    for blk in &blocks[..pre] {
        voiced.push(rms_db(&b.runner.push(&mut cont, blk).unwrap()));
        b.runner.push(&mut rolled, blk).unwrap();
    }
    let mut during = Vec::new();
    for blk in &blocks[pre..pre + quiet] {
        during.push(rms_db(&b.runner.push(&mut cont, blk).unwrap()));
        rolled.roll(blk).unwrap();
    }
    let (mut after_cont, mut after_roll) = (Vec::new(), Vec::new());
    for blk in &blocks[pre + quiet..] {
        after_cont.push(rms_db(&b.runner.push(&mut cont, blk).unwrap()));
        after_roll.push(rms_db(&b.runner.push(&mut rolled, blk).unwrap()));
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let voiced_level = mean(&voiced[voiced.len() - 4..]);
    eprintln!(
        "voiced output {voiced_level:.1} dBFS; continuous through the pause {:?} dBFS; after the pause continuous {:?} / rolled {:?} dBFS",
        during.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(),
        after_cont.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(),
        after_roll.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
    );
    for (k, (c, r)) in after_cont.iter().zip(&after_roll).enumerate() {
        assert!(r <= &(c + 3.0), "block {k} after the pause: rolled {r:.1} dBFS louder than continuous {c:.1} dBFS");
    }
}

#[test]
fn roll_against_the_graphs() {
    let Some(mut b) = bench() else { return };
    rolled_stream_matches_continuous_pushes(&mut b);
    rolled_pause_leaves_no_stale_speech(&mut b);
}
