//! Pure-Rust signal processing used around the neural stages.
//!
//! Every routine here reproduces a specific reference implementation bit-for-bit (within float
//! rounding) so that the neural graphs see the same inputs they were trained and validated with:
//!
//! * `resample`: `torchaudio.functional.resample` (windowed-sinc, Hann, width 6, roll-off 0.99)
//! * `mel`: the 22.05 kHz / 1024 / 256 / 80-bin log-mel used by seed-vc (`modules/audio.py`)
//! * `fbank`: `torchaudio.compliance.kaldi.fbank` with the CAM++ defaults
//! * `hift`: HiFT's harmonic source, 16-point STFT and ISTFT
//! * `sola`: block stitching by synchronous overlap-add
//! * `vad`: energy gate with adaptive floor and hysteresis
//!
//! Parity tests live in `tests/` and read the fixtures written by `mimictrain.export` into `<set>/fixtures`.

// index loops over small fixed-size buffers are the clearest form here and compile to the same code
#![allow(clippy::needless_range_loop)]

pub mod fbank;
pub mod hift;
pub mod mel;
pub mod noise;
pub mod resample;
pub mod sola;
pub mod vad;
pub mod window;

pub use fbank::KaldiFbank;
pub use mel::MelSpec;
pub use resample::SincResampler;
pub use sola::Sola;
pub use vad::EnergyVad;
