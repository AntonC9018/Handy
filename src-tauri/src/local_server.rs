use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{
    get_settings, write_settings, ModelUnloadTimeout, OrtAcceleratorSetting,
    TranscribeAcceleratorSetting,
};
use crate::CliArgs;
use handy_local_server::{Backend, ControlRequest, LocalServer, ModelRequest, Segment, Transcription};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};

struct HandyBackend {
    app: AppHandle,
    models: Arc<ModelManager>,
    transcription: Arc<TranscriptionManager>,
    selected_model: Mutex<String>,
    device_index: Option<usize>,
}

impl Backend for HandyBackend {
    fn status(&self) -> Value {
        let selected = self.selected_model.lock().ok().map(|id| id.clone());
        let model = self
            .models
            .get_available_models()
            .into_iter()
            .find(|model| Some(&model.id) == selected.as_ref());
        let settings = get_settings(&self.app);
        let native_model = model.as_ref().is_some_and(|model| {
            matches!(
                model.engine_type,
                crate::managers::model::EngineType::TranscribeCpp
            )
        });
        let acceleration = if !native_model {
            if settings.ort_accelerator == OrtAcceleratorSetting::Cpu {
                "cpu".into()
            } else {
                "auto".into()
            }
        } else {
            match settings.transcribe_accelerator {
                TranscribeAcceleratorSetting::Cpu => "cpu".to_string(),
                TranscribeAcceleratorSetting::Gpu => settings
                    .transcribe_gpu_device
                    .as_ref()
                    .map(|id| format!("gpu:{id}"))
                    .unwrap_or_else(|| "auto".into()),
                TranscribeAcceleratorSetting::Auto => "auto".into(),
            }
        };
        let unload_timeout = match settings.model_unload_timeout {
            ModelUnloadTimeout::Never => "never",
            ModelUnloadTimeout::Immediately => "immediately",
            ModelUnloadTimeout::Min2 => "min_2",
            ModelUnloadTimeout::Min5 => "min_5",
            ModelUnloadTimeout::Min10 => "min_10",
            ModelUnloadTimeout::Min15 => "min_15",
            ModelUnloadTimeout::Hour1 => "hour_1",
            ModelUnloadTimeout::Sec15 => "immediately",
        };
        json!({
            "model_id": selected,
            "model": model,
            "ready": model.as_ref().is_some_and(|model| model.is_downloaded),
            "acceleration": acceleration,
            "model_unload_timeout": unload_timeout,
            "gpu_devices": if native_model { crate::managers::transcription::get_available_accelerators().gpu_devices } else { vec![] },
        })
    }

    fn models(&self) -> Value {
        json!({"models": self.models.get_available_models()})
    }

    fn select_model(&self, request: ModelRequest) -> Result<Value, String> {
        if !self
            .models
            .get_available_models()
            .iter()
            .any(|model| model.id == request.model_id)
        {
            return Err("Unknown Handy model.".into());
        }
        if request.download {
            tauri::async_runtime::block_on(self.models.download_model(&request.model_id))
                .map_err(|_| "Handy model download failed.")?;
        }
        self.transcription
            .load_model_with_device(&request.model_id, self.device_index)
            .map_err(|_| "Handy model load failed.")?;
        *self
            .selected_model
            .lock()
            .map_err(|_| "Handy model selection is unavailable.")? = request.model_id.clone();
        let mut settings = get_settings(&self.app);
        settings.selected_model = request.model_id;
        write_settings(&self.app, settings);
        Ok(self.models())
    }

    fn control(&self, request: ControlRequest) -> Result<Value, String> {
        match request {
            ControlRequest::CancelDownload { model_id } => {
                self.models
                    .cancel_download(&model_id)
                    .map_err(|_| "Could not cancel model download.")?;
                return Ok(self.models());
            }
            ControlRequest::RemoveModel { model_id } => {
                if self.models.get_model_info(&model_id).is_none() {
                    return Err("Unknown model.".into());
                }
                let mut selected = self
                    .selected_model
                    .lock()
                    .map_err(|_| "Model selection unavailable.")?;
                if *selected == model_id {
                    self.transcription
                        .unload_model()
                        .map_err(|_| "Could not unload model.")?;
                    *selected = String::new();
                    let mut settings = get_settings(&self.app);
                    settings.selected_model.clear();
                    write_settings(&self.app, settings);
                }
                self.models
                    .delete_model(&model_id)
                    .map_err(|_| "Could not remove model.")?;
                return Ok(self.models());
            }
            ControlRequest::Acceleration { value } => {
                let mut settings = get_settings(&self.app);
                match value.as_str() {
                    "auto" => {
                        settings.transcribe_accelerator = TranscribeAcceleratorSetting::Auto;
                        settings.ort_accelerator = OrtAcceleratorSetting::Auto;
                    }
                    "cpu" => {
                        settings.transcribe_accelerator = TranscribeAcceleratorSetting::Cpu;
                        settings.ort_accelerator = OrtAcceleratorSetting::Cpu;
                    }
                    _ => {
                        let id = value.strip_prefix("gpu:").ok_or("Invalid acceleration.")?;
                        if !crate::managers::transcription::get_available_accelerators()
                            .gpu_devices
                            .iter()
                            .any(|device| device.id == id)
                        {
                            return Err("Unknown GPU.".into());
                        }
                        // ONNX engines have their own accelerator policy; a
                        // specific ggml device cannot select an ONNX device.
                        let selected = self
                            .selected_model
                            .lock()
                            .map_err(|_| "Model selection unavailable.")?;
                        let model = self
                            .models
                            .get_model_info(&selected)
                            .ok_or("Select a model first.")?;
                        if !matches!(
                            model.engine_type,
                            crate::managers::model::EngineType::TranscribeCpp
                        ) {
                            return Err("This model supports Auto or CPU acceleration.".into());
                        }
                        settings.transcribe_accelerator = TranscribeAcceleratorSetting::Gpu;
                        settings.transcribe_gpu_device = Some(id.into());
                    }
                }
                self.transcription
                    .unload_model()
                    .map_err(|_| "Could not unload model.")?;
                write_settings(&self.app, settings);
                crate::managers::transcription::apply_accelerator_settings(&self.app);
            }
            ControlRequest::UnloadTimeout { value } => {
                let timeout = match value.as_str() {
                    "never" => ModelUnloadTimeout::Never,
                    "immediately" => ModelUnloadTimeout::Immediately,
                    "min_2" => ModelUnloadTimeout::Min2,
                    "min_5" => ModelUnloadTimeout::Min5,
                    "min_10" => ModelUnloadTimeout::Min10,
                    "min_15" => ModelUnloadTimeout::Min15,
                    "hour_1" => ModelUnloadTimeout::Hour1,
                    _ => return Err("Invalid unload timeout.".into()),
                };
                let mut settings = get_settings(&self.app);
                settings.model_unload_timeout = timeout;
                write_settings(&self.app, settings);
                if timeout == ModelUnloadTimeout::Immediately {
                    self.transcription
                        .unload_model()
                        .map_err(|_| "Could not unload model.")?;
                }
            }
        }
        Ok(self.status())
    }

    fn transcribe(&self, pcm: Vec<f32>, language: &str, translate: bool) -> Result<Transcription, String> {
        let model = self
            .selected_model
            .lock()
            .map_err(|_| "Handy model selection is unavailable.")?
            .clone();
        // Honors Handy's idle-unload policy while keeping the engine resident
        // between requests when configured. No new process per recording.
        if !self.transcription.is_model_loaded() {
            self.transcription
                .load_model_with_device(&model, self.device_index)
                .map_err(|_| "Handy model load failed.")?;
        }
        // The request PCM is 16 kHz mono, like Handy's own recordings.
        let duration_secs = pcm.len() as f64 / 16_000.0;
        let mut result = self
            .transcription
            .transcribe_timed_with_options(pcm, Some((language, translate)))
            .map_err(|_| "Handy transcription failed.".to_string())?;
        // Engines without segment timings still get one row spanning the whole
        // request, so timing-based clients (FilmCraft) have bounds to place words in.
        if result.segments.is_empty() && !result.text.is_empty() {
            result.segments.push(crate::managers::transcription::TimedSegment {
                start: 0.0,
                end: duration_secs,
                text: result.text.clone(),
            });
        }
        Ok(Transcription {
            text: result.text,
            segments: result
                .segments
                .into_iter()
                .map(|s| Segment { start: s.start, end: s.end, text: s.text })
                .collect(),
        })
    }
}

pub(crate) fn run(app: &AppHandle, args: &CliArgs) -> i32 {
    let Some(port) = args.serve else {
        return 2;
    };
    let token = std::env::var("HANDY_SERVER_TOKEN").unwrap_or_default();
    let server = match LocalServer::bind(port, &token) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("error: {error}");
            return 2;
        }
    };
    let models = app.state::<Arc<ModelManager>>().inner().clone();
    let transcription = app.state::<Arc<TranscriptionManager>>().inner().clone();
    let selected = args.model.clone().unwrap_or_else(|| {
        let selected = get_settings(app).selected_model;
        if selected.is_empty() {
            "parakeet-tdt-0.6b-v3".into()
        } else {
            selected
        }
    });
    let backend = Arc::new(HandyBackend {
        app: app.clone(),
        models,
        transcription,
        selected_model: Mutex::new(selected.clone()),
        device_index: args.device_index,
    });
    if args.download_model {
        if let Err(error) = backend.select_model(ModelRequest {
            model_id: selected,
            download: true,
        }) {
            eprintln!("error: {error}");
            return 1;
        }
    }
    // Startup with no model is allowed: T3 can list/download/select one.
    eprintln!("Handy local server listening on 127.0.0.1:{port}");
    server.serve(backend);
    0
}
