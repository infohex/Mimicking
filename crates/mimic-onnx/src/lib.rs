//! ONNX Runtime backend.
//!
//! [`OnnxRuntime`] owns the ORT environment (the dylib is loaded at runtime from `ORT_DYLIB_PATH` or an
//! explicit path, so no link-time dependency on a particular onnxruntime build). [`OnnxGraph`] wraps one
//! static-shape session and implements [`mimic_core::Graph`].
//!
//! All graphs of a model set have float32 inputs and outputs (the exporter checks this); fp16 is internal to the graph.

use mimic_core::{Error, Graph, Result};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod vad;
pub use vad::SileroVad;

fn ort_err<R>(e: ort::Error<R>) -> Error {
    Error::Backend(e.to_string())
}

/// Where to run graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA execution provider on the given device ordinal.
    Cuda(i32),
}

/// Platform file name of the ONNX Runtime shared library.
pub fn dylib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// The process's one ONNX Runtime environment (see [`OnnxRuntime::load`]).
static ENV: std::sync::OnceLock<Arc<Environment>> = std::sync::OnceLock::new();
static ENV_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serialises CUDA Graph capture across threads. While one thread captures (the second `run_binding` of a
/// bound graph), CUDA calls that are not capture-safe on *any other thread of the process* — allocations,
/// synchronous copies, even on another device — fail with error 901 ("operation failed due to a previous
/// error during capture") and the capture itself is invalidated. Loading several model sets from several
/// threads at once is exactly that, so every `BoundGraph::load` + `warm_up` runs under this lock. Replaying
/// captured graphs from several threads is fine: replay does not capture.
pub(crate) static CAPTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Process-wide ONNX Runtime environment. Create once, share by clone.
#[derive(Clone)]
pub struct OnnxRuntime {
    env: Arc<Environment>,
    device: Device,
    cuda_graph: bool,
}

impl OnnxRuntime {
    /// Load `onnxruntime.dll` (or `.so`/`.dylib`) from `dylib`. Execution-provider libraries
    /// (`onnxruntime_providers_cuda.*`, `onnxruntime_providers_shared.*`) must sit next to it, and the
    /// CUDA / cuDNN runtime libraries must be discoverable (on Windows: on `PATH`).
    pub fn load(dylib: impl AsRef<Path>, device: Device) -> Result<Self> {
        // ONNX Runtime allows exactly one environment per process. The first call creates it; every later
        // call (another device, a reload, a second pool) gets the same one, whatever `dylib` says: the
        // library is already mapped and cannot be swapped without restarting the process.
        let _guard = ENV_INIT.lock().unwrap_or_else(|p| p.into_inner());
        let env = match ENV.get() {
            Some(e) => e.clone(),
            None => {
                let env = ort::init_from(dylib.as_ref())
                    .map_err(|e| {
                        Error::Backend(format!(
                            "loading onnxruntime from {}: {e}",
                            dylib.as_ref().display()
                        ))
                    })?
                    .with_name("mimicking")
                    .build()
                    .map_err(ort_err)?;
                let env = Arc::new(env);
                let _ = ENV.set(env.clone());
                env
            }
        };
        Ok(Self {
            env,
            device,
            cuda_graph: false,
        })
    }

    /// Same as [`load`](Self::load) but takes the library path from `ORT_DYLIB_PATH`.
    pub fn from_env(device: Device) -> Result<Self> {
        let p = std::env::var_os("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .ok_or_else(|| Error::Backend("ORT_DYLIB_PATH is not set".into()))?;
        Self::load(p, device)
    }

    /// Locate and load ONNX Runtime following the documented runtime layout:
    ///
    /// 1. `explicit`, when given
    /// 2. `$MIMIC_ORT_DIR/<dylib name>`
    /// 3. `<exe dir>/runtime/onnxruntime/<dylib name>`
    /// 4. `$ORT_DYLIB_PATH`
    ///
    /// Before loading, `<exe dir>/runtime/cuda` and `$MIMIC_CUDA_DIR` are prepended to `PATH` so the CUDA
    /// execution provider finds cudart / cuBLAS / cuDNN there.
    pub fn discover(explicit: Option<&Path>, device: Device) -> Result<Self> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        // CUDA runtime libraries on PATH (LoadLibrary reads the current value)
        let mut extra: Vec<PathBuf> = Vec::new();
        if let Some(d) = std::env::var_os("MIMIC_CUDA_DIR") {
            extra.push(PathBuf::from(d));
        }
        if let Some(e) = &exe_dir {
            let d = e.join("runtime").join("cuda");
            if d.is_dir() {
                extra.push(d);
            }
        }
        if !extra.is_empty() {
            let mut paths: Vec<PathBuf> = extra.clone();
            if let Some(cur) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&cur));
            }
            if let Ok(joined) = std::env::join_paths(paths) {
                // SAFETY: called before any thread that reads PATH is spawned by this library
                unsafe { std::env::set_var("PATH", joined) };
            }
        }
        let name = dylib_name();
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = explicit {
            candidates.push(p.to_path_buf());
        }
        if let Some(d) = std::env::var_os("MIMIC_ORT_DIR") {
            candidates.push(PathBuf::from(d).join(name));
        }
        if let Some(e) = &exe_dir {
            candidates.push(e.join("runtime").join("onnxruntime").join(name));
        }
        if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
            candidates.push(PathBuf::from(p));
        }
        let Some(found) = candidates.iter().find(|p| p.is_file()) else {
            return Err(Error::Backend(format!(
                "onnxruntime not found; looked at: {}",
                candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        tracing::info!(path = %found.display(), "onnxruntime");
        Self::load(found, device)
    }

    /// Ask the CUDA EP to capture/replay the whole graph (requires every node on the EP and fixed I/O addresses).
    /// Off by default; enable per runtime once device-resident I/O binding is in place.
    /// Same environment, different device for the graphs loaded from now on. ONNX Runtime allows one
    /// environment per process, so a host that wants to retry on the CPU, or reload models later, must reuse
    /// the runtime it already has instead of calling [`load`](Self::load) again.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_cuda_graph(mut self, enable: bool) -> Self {
        self.cuda_graph = enable;
        self
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Plain session (any shapes, may fall back to CPU for shape ops); never asks for CUDA Graph capture.
    pub fn load_graph(&self, path: impl AsRef<Path>) -> Result<OnnxGraph> {
        OnnxGraph::load(self, path.as_ref(), false)
    }
}

/// One static-shape ONNX session.
pub struct OnnxGraph {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    path: PathBuf,
    /// inputs given through `set_input`, consumed by `run_set`
    cached: Vec<Option<ArrayD<f32>>>,
    /// outputs of the last `run_set_keep`
    last_outputs: Vec<ArrayD<f32>>,
}

impl OnnxGraph {
    fn load(rt: &OnnxRuntime, path: &Path, cuda_graph: bool) -> Result<Self> {
        let mut builder = Session::builder(&rt.env).map_err(ort_err)?;
        match rt.device {
            Device::Cpu => {}
            Device::Cuda(id) => {
                let ep = ort::ep::CUDA::default()
                    .with_device_id(id)
                    .with_cuda_graph(cuda_graph)
                    .build()
                    .error_on_failure();
                builder = builder.with_execution_providers([ep]).map_err(ort_err)?;
            }
        }
        // optimisation level is left at ORT's default (enable all): the crate's newest level constant is
        // rejected by onnxruntime 1.22.
        // Thread pools: two intra-op threads are plenty (the CUDA EP leaves almost nothing on the CPU), and
        // spinning is off so idle pool threads block instead of burning cores between runs. With seven sessions
        // per pipeline, spinning pools starve a host's audio threads and show up as capture overruns.
        let session = builder
            .with_intra_threads(2)
            .map_err(ort_err)?
            .with_config_entry("session.intra_op.allow_spinning", "0")
            .map_err(ort_err)?
            .with_config_entry("session.inter_op.allow_spinning", "0")
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(|e| Error::Backend(format!("loading {}: {e}", path.display())))?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        tracing::info!(graph = %path.display(), ?input_names, ?output_names, "loaded onnx graph");
        let cached: Vec<Option<ArrayD<f32>>> = vec![None; input_names.len()];
        Ok(Self {
            session,
            input_names,
            output_names,
            path: path.to_path_buf(),
            cached,
            last_outputs: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ------------------------------------------------------------------------------------------------ device-resident

fn cuda_err(code: cudarc::runtime::sys::cudaError_t, what: &str) -> Result<()> {
    if code == cudarc::runtime::sys::cudaError_t::cudaSuccess {
        Ok(())
    } else {
        Err(Error::Backend(format!("{what}: cuda error {code:?}")))
    }
}

/// A device buffer owned by ORT (so it can be bound) with host copy helpers.
struct DeviceBuf {
    tensor: Tensor<f32>,
    len: usize,
    shape: Vec<usize>,
}

impl DeviceBuf {
    fn new(alloc: &ort::memory::Allocator, shape: &[usize]) -> Result<Self> {
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let tensor = Tensor::<f32>::new(alloc, dims).map_err(ort_err)?;
        Ok(Self {
            tensor,
            len: shape.iter().product(),
            shape: shape.to_vec(),
        })
    }

    fn upload(&mut self, src: &[f32]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Shape(format!(
                "upload: {} values into buffer of {} ({:?})",
                src.len(),
                self.len,
                self.shape
            )));
        }
        let dst = self.tensor.data_ptr_mut();
        let bytes = std::mem::size_of_val(src);
        // cudaMemcpyDefault relies on unified addressing: works whatever the current device is
        let code = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                dst,
                src.as_ptr() as *const _,
                bytes,
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDefault,
            )
        };
        cuda_err(code, "host->device copy")
    }

    /// Device-to-device copy from `src` (same element count).
    fn copy_from_device(&mut self, src: &DeviceBuf) -> Result<()> {
        if src.len != self.len {
            return Err(Error::Shape(format!(
                "device copy: {} -> {} elements",
                src.len, self.len
            )));
        }
        let code = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                self.tensor.data_ptr_mut(),
                src.tensor.data_ptr(),
                self.len * std::mem::size_of::<f32>(),
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDefault,
            )
        };
        cuda_err(code, "device->device copy")
    }

    fn download(&self, dst: &mut [f32]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Shape(format!(
                "download: buffer {} vs {}",
                dst.len(),
                self.len
            )));
        }
        let src = self.tensor.data_ptr();
        let bytes = std::mem::size_of_val(dst);
        let code = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                dst.as_mut_ptr() as *mut _,
                src,
                bytes,
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDefault,
            )
        };
        cuda_err(code, "device->host copy")
    }
}

/// A static-shape session with every input and output bound to fixed device buffers.
///
/// Running means: copy inputs host->device, `run_binding`, copy outputs device->host. Because addresses
/// never change, the CUDA EP can capture the whole graph once and replay it (`with_cuda_graph(true)` on the
/// runtime). ORT captures on the second run and replays from the third, so [`BoundGraph::warm_up`] runs twice.
pub struct BoundGraph {
    session: Session,
    binding: ort::session::IoBinding,
    inputs: Vec<DeviceBuf>,
    outputs: Vec<DeviceBuf>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    path: PathBuf,
}

impl BoundGraph {
    /// `input_shapes` / `output_shapes` in the graph's declared order.
    pub fn load(
        rt: &OnnxRuntime,
        path: &Path,
        input_shapes: &[Vec<usize>],
        output_shapes: &[Vec<usize>],
    ) -> Result<Self> {
        let Device::Cuda(device_id) = rt.device else {
            return Err(Error::Backend("BoundGraph needs a CUDA device".into()));
        };
        // session creation allocates on the device: not safe while another thread captures
        let _capture = CAPTURE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let g = OnnxGraph::load(rt, path, rt.cuda_graph)?;
        let OnnxGraph {
            session,
            input_names,
            output_names,
            path,
            ..
        } = g;
        if input_shapes.len() != input_names.len() || output_shapes.len() != output_names.len() {
            return Err(Error::Shape(format!(
                "{}: graph has {} inputs / {} outputs, got {} / {} shapes",
                path.display(),
                input_names.len(),
                output_names.len(),
                input_shapes.len(),
                output_shapes.len()
            )));
        }
        let mem = ort::memory::MemoryInfo::new(
            ort::memory::AllocationDevice::CUDA,
            device_id,
            ort::memory::AllocatorType::Device,
            ort::memory::MemoryType::Default,
        )
        .map_err(ort_err)?;
        let alloc = ort::memory::Allocator::new(&session, mem).map_err(ort_err)?;
        let mut inputs = Vec::new();
        for s in input_shapes {
            inputs.push(DeviceBuf::new(&alloc, s)?);
        }
        let mut outputs = Vec::new();
        for s in output_shapes {
            outputs.push(DeviceBuf::new(&alloc, s)?);
        }
        let mut binding = session.create_binding().map_err(ort_err)?;
        for (n, b) in input_names.iter().zip(&inputs) {
            binding.bind_input(n.as_str(), &b.tensor).map_err(ort_err)?;
        }
        for (n, b) in output_names.iter().zip(outputs.iter_mut()) {
            binding
                .bind_output(n.as_str(), &mut b.tensor)
                .map_err(ort_err)?;
        }
        Ok(Self {
            session,
            binding,
            inputs,
            outputs,
            input_names,
            output_names,
            path,
        })
    }

    /// Run the graph twice on whatever is in the buffers (zeros after load) so capture happens before real use.
    /// Holds the process-wide capture lock: do this before real-time work starts, and never while other
    /// threads are running graphs (their copies would land inside the capture window).
    pub fn warm_up(&mut self) -> Result<()> {
        let _capture = CAPTURE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for _ in 0..2 {
            self.session.run_binding(&self.binding).map_err(ort_err)?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn input_shape(&self, i: usize) -> &[usize] {
        &self.inputs[i].shape
    }

    pub fn output_shape(&self, i: usize) -> &[usize] {
        &self.outputs[i].shape
    }
}

impl Graph for BoundGraph {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, inputs: &[ArrayViewD<'_, f32>]) -> Result<Vec<ArrayD<f32>>> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::Shape(format!(
                "{}: expected {} inputs, got {}",
                self.path.display(),
                self.inputs.len(),
                inputs.len()
            )));
        }
        for (buf, a) in self.inputs.iter_mut().zip(inputs) {
            if a.shape() != buf.shape.as_slice() {
                return Err(Error::Shape(format!(
                    "{}: input shape {:?}, bound {:?}",
                    self.path.display(),
                    a.shape(),
                    buf.shape
                )));
            }
            match a.as_slice() {
                Some(s) => buf.upload(s)?,
                None => {
                    let owned = a.to_owned();
                    buf.upload(owned.as_slice().unwrap())?;
                }
            }
        }
        self.run_set()
    }

    fn set_input(&mut self, index: usize, a: ArrayViewD<'_, f32>) -> Result<()> {
        let buf = self
            .inputs
            .get_mut(index)
            .ok_or_else(|| Error::Shape(format!("{}: no input {index}", self.path.display())))?;
        if a.shape() != buf.shape.as_slice() {
            return Err(Error::Shape(format!(
                "{}: input {index} shape {:?}, bound {:?}",
                self.path.display(),
                a.shape(),
                buf.shape
            )));
        }
        match a.as_slice() {
            Some(s) => buf.upload(s),
            None => buf.upload(a.to_owned().as_slice().unwrap()),
        }
    }

    fn run_set(&mut self) -> Result<Vec<ArrayD<f32>>> {
        self.run_set_keep()?;
        (0..self.outputs.len())
            .map(|j| self.get_output(j))
            .collect()
    }

    fn run_set_keep(&mut self) -> Result<()> {
        self.session
            .run_binding(&self.binding)
            .map_err(ort_err)
            .map(|_| ())
    }

    fn get_output(&mut self, index: usize) -> Result<ArrayD<f32>> {
        let buf = self
            .outputs
            .get(index)
            .ok_or_else(|| Error::Shape(format!("{}: no output {index}", self.path.display())))?;
        let mut v = vec![0.0f32; buf.len];
        buf.download(&mut v)?;
        Ok(ArrayD::from_shape_vec(IxDyn(&buf.shape), v).unwrap())
    }

    fn copy_output_to_input(&mut self, output: usize, input: usize) -> Result<()> {
        if output >= self.outputs.len() || input >= self.inputs.len() {
            return Err(Error::Shape(format!(
                "{}: copy output {output} -> input {input} out of range",
                self.path.display()
            )));
        }
        let (ins, outs) = (&mut self.inputs, &self.outputs);
        ins[input].copy_from_device(&outs[output])
    }
}

impl Graph for OnnxGraph {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, inputs: &[ArrayViewD<'_, f32>]) -> Result<Vec<ArrayD<f32>>> {
        if inputs.len() != self.input_names.len() {
            return Err(Error::Shape(format!(
                "{}: expected {} inputs, got {}",
                self.path.display(),
                self.input_names.len(),
                inputs.len()
            )));
        }
        let mut values: Vec<(String, ort::value::DynValue)> = Vec::with_capacity(inputs.len());
        for (name, a) in self.input_names.iter().zip(inputs) {
            let owned = a.to_owned(); // standard layout copy; ORT copies to device anyway
            let t = Tensor::from_array(owned).map_err(ort_err)?;
            values.push((name.clone(), t.into_dyn()));
        }
        let outputs = self.session.run(values).map_err(ort_err)?;
        let mut result = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let v = outputs
                .get(name)
                .ok_or_else(|| Error::Backend(format!("missing output {name}")))?;
            let arr: ArrayViewD<f32> = v.try_extract_array::<f32>().map_err(ort_err)?;
            result.push(arr.to_owned().into_dimensionality::<IxDyn>().unwrap());
        }
        Ok(result)
    }

    fn set_input(&mut self, index: usize, value: ArrayViewD<'_, f32>) -> Result<()> {
        let slot = self
            .cached
            .get_mut(index)
            .ok_or_else(|| Error::Shape(format!("{}: no input {index}", self.path.display())))?;
        *slot = Some(value.to_owned());
        Ok(())
    }

    fn run_set(&mut self) -> Result<Vec<ArrayD<f32>>> {
        let taken: Vec<ArrayD<f32>> = self
            .cached
            .iter()
            .enumerate()
            .map(|(i, c)| {
                c.clone().ok_or_else(|| {
                    Error::Shape(format!("{}: input {i} was never set", self.path.display()))
                })
            })
            .collect::<Result<_>>()?;
        let views: Vec<ArrayViewD<f32>> = taken.iter().map(|a| a.view()).collect();
        self.run(&views)
    }

    fn run_set_keep(&mut self) -> Result<()> {
        self.last_outputs = self.run_set()?;
        Ok(())
    }

    fn get_output(&mut self, index: usize) -> Result<ArrayD<f32>> {
        self.last_outputs.get(index).cloned().ok_or_else(|| {
            Error::Shape(format!(
                "{}: no output {index} from the last run",
                self.path.display()
            ))
        })
    }

    fn copy_output_to_input(&mut self, output: usize, input: usize) -> Result<()> {
        let v = self.get_output(output)?;
        self.set_input(input, v.view())
    }
}
