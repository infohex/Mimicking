//! `Session::roll` without a model: the analysis window, the gate queue and the counters behave like `push`
//! would, and the shape check matches. The model-backed parity test (rolled vs continuous pushes) lives in
//! `mimic-model/tests/roll.rs`, next to the graphs it needs.

use mimic_core::Prompt;
use mimic_dsp::EnergyVad;
use mimic_pipeline::{EulerCfg, Session, StreamGeometry};
use ndarray::{Array2, Array3};

/// A small geometry with the shape of the 2.0 sets (block 160 ms, right context 20 ms at 22.05 k).
fn geo() -> StreamGeometry {
    StreamGeometry {
        acoustic_sr: 22050,
        content_sr: 16000,
        block_frame: 3528,
        sola_buffer_frame: 882,
        sola_search_frame: 441,
        extra_frame_right: 441,
        window_16k: 43840,
        block_16k: 2560,
        skip_tail: 1,
        return_length: 11,
        ce_dit_frame_difference: 100,
        t_cond: 63,
        prompt_samples: 66150,
        prompt_frames: 258,
    }
}

fn prompt() -> Prompt {
    Prompt {
        cond: Array3::zeros((1, 258, 4)),
        mel: Array3::zeros((1, 80, 258)),
        style: Array2::zeros((1, 2)),
    }
}

fn session(vad: bool) -> Session {
    let s = Session::new(geo(), prompt(), EulerCfg { steps: 1, cfg: 0.0 }, 1);
    if vad {
        s.with_vad(Box::new(EnergyVad::default()))
    } else {
        s
    }
}

#[test]
fn roll_keeps_the_real_audio_in_the_window() {
    let g = geo();
    let mut s = session(false);
    assert!(s.window16().iter().all(|&v| v == 0.0));
    // 20 blocks of a ramp; the window holds the last `window_16k` samples of what was rolled in
    let n_blocks = 20;
    let signal: Vec<f32> = (0..n_blocks * g.block_16k).map(|i| i as f32 * 1e-6).collect();
    for b in signal.chunks(g.block_16k) {
        s.roll(b).unwrap();
    }
    let expect = &signal[signal.len() - g.window_16k..];
    assert_eq!(s.window16(), expect);
    assert_eq!(s.rolled, n_blocks);
    assert_eq!(s.blocks, 0);
    // fewer blocks than the window: the leading part is still the initial silence
    let mut s = session(false);
    for b in signal.chunks(g.block_16k).take(3) {
        s.roll(b).unwrap();
    }
    let w = s.window16();
    assert!(w[..g.window_16k - 3 * g.block_16k].iter().all(|&v| v == 0.0));
    assert_eq!(&w[g.window_16k - 3 * g.block_16k..], &signal[..3 * g.block_16k]);
}

#[test]
fn roll_checks_the_block_length_like_push() {
    let g = geo();
    let mut s = session(false);
    assert!(s.roll(&vec![0.0; g.block_16k - 1]).is_err());
    assert!(s.roll(&vec![0.0; g.block_16k + 1]).is_err());
    assert!(s.roll(&vec![0.0; g.block_16k]).is_ok());
}

#[test]
fn roll_keeps_the_gate_queue_length() {
    let g = geo();
    let mut s = session(true);
    let before = s.gate_delay_blocks();
    for _ in 0..7 {
        s.roll(&vec![0.0; g.block_16k]).unwrap();
    }
    assert_eq!(s.gate_delay_blocks(), before);
    let mut loud = session(true);
    let tone: Vec<f32> = (0..g.block_16k).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
    for _ in 0..7 {
        loud.roll(&tone).unwrap();
    }
    assert_eq!(loud.gate_delay_blocks(), before);
    assert_eq!(loud.rolled, 7);
}
