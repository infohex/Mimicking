//! Convert `source.wav` to the voice of `reference.wav` in as few lines as the library allows.
//!
//!     cargo run --release -p convert-minimal -- reference.wav source.wav out.wav
//!
//! Needs `ORT_DYLIB_PATH` (onnxruntime.dll) and a model set in `models/Mimicking-v2.0-lowlat` (see README: "Models and runtime libraries").

use anyhow::Result;
use mimic_pipeline::{Converter, EulerCfg, Runner, Session};
use mimic_model::{Device, ModelFiles, OnnxRuntime, WindowConfig, ModelSet, CONTENT_SR, SR};

fn read_mono(path: &str, target_sr: usize) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
    let ch = spec.channels as usize;
    let mono: Vec<f32> = r
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 * scale)
        .collect::<Vec<_>>()
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok(mimic_dsp::SincResampler::new(spec.sample_rate as usize, target_sr).resample(&mono))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (reference, source, out) = (&args[1], &args[2], &args[3]);

    let rt = OnnxRuntime::from_env(Device::Cuda(0))?.with_cuda_graph(true);
    let model = ModelSet::load(
        &rt,
        &ModelFiles::new("models/Mimicking-v2.0-lowlat"),
        WindowConfig::default(),
    )?;
    let (stages, geo) = model.into_stages();
    let mut runner = Runner::new(stages);

    let prompt = runner
        .stages
        .prepare_prompt(&read_mono(reference, SR)?, &geo)?;
    let mut session = Session::new(
        geo,
        prompt,
        EulerCfg {
            steps: 10,
            cfg: 0.7,
        },
        1234,
    );
    let y = Converter::convert(
        &mut runner,
        &mut session,
        &read_mono(source, CONTENT_SR)?,
        |_, _| {},
    )?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(out, spec)?;
    for v in y {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
