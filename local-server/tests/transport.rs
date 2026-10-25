use handy_local_server::{
    Backend, ControlRequest, LocalServer, ModelRequest, Segment, Transcription, MAX_AUDIO_BYTES,
};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const TOKEN: &str = "testOnlyToken1234567890123456789012";

#[derive(Default)]
struct TestBackend {
    entered: Option<mpsc::Sender<()>>,
    release: Option<Mutex<mpsc::Receiver<()>>>,
}

impl Backend for TestBackend {
    fn status(&self) -> Value {
        json!({"ready": true})
    }
    fn models(&self) -> Value {
        json!({"models": []})
    }
    fn select_model(&self, request: ModelRequest) -> Result<Value, String> {
        Ok(json!({"selected": request.model_id, "download": request.download}))
    }
    fn control(&self, request: ControlRequest) -> Result<Value, String> {
        match request {
            ControlRequest::CancelDownload { model_id } => Ok(json!({"cancelled": model_id})),
            _ => Ok(json!({"controlled": true})),
        }
    }
    fn transcribe(&self, pcm: Vec<f32>, language: &str, translate: bool) -> Result<Transcription, String> {
        assert_eq!(pcm, vec![0.25, -0.5]);
        if let Some(entered) = &self.entered {
            entered.send(()).unwrap();
            self.release
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
        }
        Ok(Transcription {
            text: format!("{language}:{translate}"),
            segments: vec![Segment { start: 0.0, end: 0.000125, text: "timed".into() }],
        })
    }
}

#[test]
fn validates_controls_and_allows_cancellation_while_inference_is_busy() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let port = server(TestBackend {
        entered: Some(entered_tx),
        release: Some(Mutex::new(release_rx)),
    });
    let active = std::thread::spawn(move || {
        request(port, "POST", "/transcribe", &audio_headers(), &audio())
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let headers = format!("{}Content-Type: application/json\r\n", auth());
    assert_eq!(
        request(
            port,
            "POST",
            "/control",
            &headers,
            br#"{"action":"cancel_download","model_id":"test-model"}"#
        )
        .0,
        200
    );
    assert_eq!(
        request(
            port,
            "POST",
            "/control",
            &headers,
            br#"{"action":"remove_model","model_id":"test-model"}"#
        )
        .0,
        409
    );
    assert_eq!(
        request(
            port,
            "POST",
            "/control",
            &headers,
            br#"{"action":"unload_timeout","value":"never","path":"/tmp/x"}"#
        )
        .0,
        400
    );
    release_tx.send(()).unwrap();
    assert_eq!(active.join().unwrap().0, 200);
    assert_eq!(
        request(
            port,
            "POST",
            "/control",
            &headers,
            br#"{"action":"unload_timeout","value":"never"}"#
        )
        .0,
        200
    );
}

fn server(backend: TestBackend) -> u16 {
    let server = LocalServer::bind(0, TOKEN).unwrap();
    let port = server.port().unwrap();
    std::thread::spawn(move || server.serve(Arc::new(backend)));
    port
}

fn request(port: u16, method: &str, path: &str, headers: &str, body: &[u8]) -> (u16, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n", body.len()).unwrap();
    stream.write_all(body).unwrap();
    let mut output = String::new();
    stream.read_to_string(&mut output).unwrap();
    let (headers, body) = output.split_once("\r\n\r\n").unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}

fn auth() -> String {
    format!("Authorization: Bearer {TOKEN}\r\n")
}

fn audio_headers() -> String {
    format!("{}Content-Type: application/octet-stream\r\nX-Speech-Language: ru\r\nX-Speech-Translate: true\r\n", auth())
}

fn audio() -> Vec<u8> {
    [0.25_f32, -0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

#[test]
fn rejects_unauthorized_browser_and_rebinding_requests() {
    let port = server(TestBackend::default());
    assert_eq!(request(port, "GET", "/status", "", &[]).0, 401);
    assert_eq!(
        request(
            port,
            "GET",
            "/status",
            "Authorization: Bearer wrong\r\n",
            &[]
        )
        .0,
        401
    );
    assert_eq!(
        request(
            port,
            "GET",
            "/status",
            &format!("{}Origin: https://example.com\r\n", auth()),
            &[]
        )
        .0,
        403
    );
    assert!(LocalServer::bind(0, "short").is_err());
}

#[test]
fn transcribes_valid_audio_with_request_options_and_rejects_malformed_audio() {
    let port = server(TestBackend::default());
    let (code, body) = request(port, "POST", "/transcribe", &audio_headers(), &audio());
    assert_eq!(code, 200);
    assert_eq!(body["text"], "ru:true");
    // segment timings ride along without changing the text-only contract
    assert_eq!(body["segments"][0]["text"], "timed");
    assert_eq!(body["segments"][0]["end"], 0.000125);
    for invalid in [
        vec![1, 2, 3],
        f32::NAN.to_le_bytes().to_vec(),
        2_f32.to_le_bytes().to_vec(),
    ] {
        assert_eq!(
            request(port, "POST", "/transcribe", &audio_headers(), &invalid).0,
            400
        );
    }
    assert!(handy_local_server::decode_pcm(&vec![0; MAX_AUDIO_BYTES + 4]).is_err());
    assert_eq!(
        request(port, "POST", "/transcribe", &auth(), &audio()).0,
        415
    );
    let headers = format!("{}Content-Type: application/json\r\n", auth());
    let (code, model) = request(
        port,
        "POST",
        "/model",
        &headers,
        br#"{"model_id":"parakeet","download":true}"#,
    );
    assert_eq!(code, 200);
    assert_eq!(model, json!({"selected": "parakeet", "download": true}));
    assert_eq!(
        request(
            port,
            "POST",
            "/model",
            &headers,
            br#"{"file":"/private.wav"}"#
        )
        .0,
        400
    );
}

#[test]
fn reports_busy_without_queueing_and_keeps_status_available() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let port = server(TestBackend {
        entered: Some(entered_tx),
        release: Some(Mutex::new(release_rx)),
    });
    let inference = std::thread::spawn(move || {
        request(port, "POST", "/transcribe", &audio_headers(), &audio())
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        request(port, "GET", "/status", &auth(), &[]).1["busy"],
        true
    );
    assert_eq!(
        request(port, "POST", "/transcribe", &audio_headers(), &audio()).0,
        409
    );
    release_tx.send(()).unwrap();
    assert_eq!(inference.join().unwrap().0, 200);
    assert_eq!(
        request(port, "GET", "/status", &auth(), &[]).1["busy"],
        false
    );
}
