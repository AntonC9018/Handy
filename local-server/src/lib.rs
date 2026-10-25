//! Authenticated loopback transport for Handy's existing transcription manager.
//! Audio stays in memory; this API never accepts client-supplied file paths.

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tiny_http::{Header, Method, Request, Response, Server};

pub const MAX_AUDIO_BYTES: usize = 16_000 * 4 * 300;
const MAX_OPTIONS_BYTES: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRequest {
    pub model_id: String,
    #[serde(default)]
    pub download: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlRequest {
    CancelDownload { model_id: String },
    RemoveModel { model_id: String },
    Acceleration { value: String },
    UnloadTimeout { value: String },
}

/// One timed stretch of a transcription reply. Times are seconds from the audio's
/// first sample; engines without timings send no segments at all.
#[derive(Debug, serde::Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A transcription: the full text and, when the engine produced them, timed segments.
#[derive(Debug, Default, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<Segment>,
}

/// Implemented by the app using its existing model and transcription managers.
pub trait Backend: Send + Sync + 'static {
    fn status(&self) -> Value;
    fn models(&self) -> Value;
    fn select_model(&self, request: ModelRequest) -> Result<Value, String>;
    fn control(&self, _request: ControlRequest) -> Result<Value, String> {
        Err("Controls are unavailable.".into())
    }
    fn transcribe(&self, pcm: Vec<f32>, language: &str, translate: bool) -> Result<Transcription, String>;
}

pub struct LocalServer {
    server: Server,
    authorization: String,
}

fn header<'a>(request: &'a Request, name: &'static str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str())
}

fn reply(request: Request, status: u16, body: Value) {
    let mut response = Response::from_string(body.to_string()).with_status_code(status);
    // Constant ASCII names and values cannot fail Header validation.
    if let Ok(header) = Header::from_bytes("Content-Type", "application/json") {
        response.add_header(header);
    }
    if let Ok(header) = Header::from_bytes("Cache-Control", "no-store") {
        response.add_header(header);
    }
    // No CORS: only the T3 backend, never the phone browser, calls this service.
    let _ = request.respond(response);
}

fn read_body(request: &mut Request, maximum: usize) -> Result<Vec<u8>, u16> {
    let length = request.body_length().ok_or(411_u16)?;
    if length == 0 || length > maximum {
        return Err(413);
    }
    let mut bytes = Vec::with_capacity(length);
    request
        .as_reader()
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| 400_u16)?;
    if bytes.len() != length {
        return Err(400);
    }
    Ok(bytes)
}

/// Decode the same five-minute, 16 kHz mono Float32 LE payload used by T3.
pub fn decode_pcm(bytes: &[u8]) -> Result<Vec<f32>, &'static str> {
    if bytes.is_empty() || bytes.len() > MAX_AUDIO_BYTES || !bytes.len().is_multiple_of(4) {
        return Err("Expected at most five minutes of 16 kHz mono Float32 LE audio.");
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if sample.is_finite() && (-1.0..=1.0).contains(&sample) {
                Ok(sample)
            } else {
                Err("Audio contains an invalid sample.")
            }
        })
        .collect()
}

struct BusyGuard(Arc<AtomicBool>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl LocalServer {
    /// Bind IPv4 loopback only. Use a separate random bearer token for this API.
    pub fn bind(port: u16, token: &str) -> Result<Self, String> {
        if token.len() < 32 || !token.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err("HANDY_SERVER_TOKEN must contain at least 32 ASCII letters/digits.".into());
        }
        let server = Server::http(("127.0.0.1", port))
            .map_err(|_| "Could not bind the Handy loopback server.".to_string())?;
        Ok(Self {
            server,
            authorization: format!("Bearer {token}"),
        })
    }

    pub fn port(&self) -> Result<u16, String> {
        self.server
            .server_addr()
            .to_ip()
            .map(|address| address.port())
            .ok_or_else(|| "Handy server is not bound to an IP address.".into())
    }

    /// Runs until shutdown. One inference/model change may run at a time;
    /// health and model queries remain available while that worker is busy.
    pub fn serve(self, backend: Arc<dyn Backend>) {
        let busy = Arc::new(AtomicBool::new(false));
        for mut request in self.server.incoming_requests() {
            // Host and Origin checks also reject browser-driven loopback access
            // and DNS rebinding. Authorization is still required for every route.
            let host = format!("127.0.0.1:{}", self.port().unwrap_or(0));
            if header(&request, "Origin").is_some()
                || header(&request, "Host") != Some(host.as_str())
            {
                reply(request, 403, json!({"error": "local_backend_only"}));
                continue;
            }
            if header(&request, "Authorization") != Some(self.authorization.as_str()) {
                reply(request, 401, json!({"error": "unauthorized"}));
                continue;
            }
            match (request.method(), request.url()) {
                (&Method::Get, "/status") => {
                    let mut status = backend.status();
                    status["busy"] = json!(busy.load(Ordering::Acquire));
                    reply(request, 200, status);
                    continue;
                }
                (&Method::Get, "/models") => {
                    reply(request, 200, backend.models());
                    continue;
                }
                (&Method::Post, "/transcribe" | "/model" | "/control") => {}
                _ => {
                    reply(request, 404, json!({"error": "not_found"}));
                    continue;
                }
            }
            let is_control = request.url() == "/control";
            if !is_control
                && busy
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                reply(request, 409, json!({"error": "speech_busy"}));
                continue;
            }
            let guard = if is_control {
                None
            } else {
                Some(BusyGuard(busy.clone()))
            };
            let is_audio = request.url() == "/transcribe";
            let content_type = if is_audio {
                "application/octet-stream"
            } else {
                "application/json"
            };
            if header(&request, "Content-Type") != Some(content_type) {
                reply(request, 415, json!({"error": "unsupported_content_type"}));
                continue;
            }
            let language = header(&request, "X-Speech-Language")
                .unwrap_or("auto")
                .to_owned();
            let translate = match header(&request, "X-Speech-Translate") {
                None | Some("false") => false,
                Some("true") => true,
                _ => {
                    reply(request, 400, json!({"error": "invalid_translation_option"}));
                    continue;
                }
            };
            if !(2..=16).contains(&language.len())
                || !language
                    .bytes()
                    .all(|byte| byte.is_ascii_alphabetic() || byte == b'-')
            {
                reply(request, 400, json!({"error": "invalid_language"}));
                continue;
            }
            let bytes = match read_body(
                &mut request,
                if is_audio {
                    MAX_AUDIO_BYTES
                } else {
                    MAX_OPTIONS_BYTES
                },
            ) {
                Ok(bytes) => bytes,
                Err(code) => {
                    reply(request, code, json!({"error": "invalid_body"}));
                    continue;
                }
            };
            let backend = backend.clone();
            if is_control {
                let options = match serde_json::from_slice::<ControlRequest>(&bytes) {
                    Ok(options) => options,
                    Err(_) => {
                        reply(request, 400, json!({"error": "invalid_control"}));
                        continue;
                    }
                };
                // Cancellation must remain available while a download is busy.
                // All other model/settings mutations share the inference lock.
                let guard = if matches!(options, ControlRequest::CancelDownload { .. }) {
                    None
                } else if busy
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    Some(BusyGuard(busy.clone()))
                } else {
                    reply(request, 409, json!({"error": "speech_busy"}));
                    continue;
                };
                let result = backend.control(options);
                drop(guard);
                match result {
                    Ok(value) => reply(request, 200, value),
                    Err(_) => reply(request, 400, json!({"error": "control_failed"})),
                }
                continue;
            }
            std::thread::spawn(move || {
                let (status, response) = if is_audio {
                    match decode_pcm(&bytes) {
                        Ok(pcm) => match backend.transcribe(pcm, &language, translate) {
                            // `segments` is additive: text-only clients keep working.
                            Ok(t) => (200, json!({"text": t.text, "segments": t.segments})),
                            Err(_) => (503, json!({"error": "transcription_unavailable"})),
                        },
                        Err(_) => (400, json!({"error": "invalid_audio"})),
                    }
                } else {
                    match serde_json::from_slice::<ModelRequest>(&bytes) {
                        Ok(options)
                            if !options.model_id.is_empty() && options.model_id.len() <= 160 =>
                        {
                            match backend.select_model(options) {
                                Ok(models) => (200, models),
                                Err(_) => (503, json!({"error": "model_unavailable"})),
                            }
                        }
                        _ => (400, json!({"error": "invalid_model_request"})),
                    }
                };
                drop(guard);
                reply(request, status, response);
            });
        }
    }
}
