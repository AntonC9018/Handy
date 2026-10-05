# Local transcription server

Run Handy's transcription engine as a service on Windows or Linux/WSL. T3 accepts
phone audio through its normal authenticated connection and forwards it to this
local server. No microphone, clipboard, paste action, or recording history is used.

Start `handy --serve 8003 --model <model-id>`. Set `HANDY_SERVER_TOKEN` to a random
string of at least 32 ASCII letters/digits in the launching process's environment.
Use `--download-model` to download that model from Handy's existing catalog first.
Without an installed model, the server starts and accepts model setup through T3.
Use `--list-models --json` to find model identifiers.

The server binds exclusively to `127.0.0.1`. Do not publish its port through
Tailscale or a reverse proxy. All requests require `Authorization: Bearer <token>`.
T3 must use the same token in `T3_SPEECH_HANDY_TOKEN` and
`T3_SPEECH_HANDY_URL=http://127.0.0.1:8003/`. Browsers call T3, not Handy directly.

The local backend protocol provides:

- `GET /status`: model metadata, readiness, busy state, acceleration, unload
  timeout, and available GPU devices for the selected engine.
- `GET /models`: Handy's model catalog and installation state.
- `POST /model`: JSON `{"model_id":"…","download":true}` downloads/selects a
  catalog model. Omit `download` to select an already installed model.
- `POST /transcribe`: `application/octet-stream`, 16 kHz mono Float32 LE PCM,
  maximum five minutes. Optional `X-Speech-Language` and `X-Speech-Translate`
  override language/translation for this request without changing Handy settings.
- `POST /control`: JSON with an `action` of `cancel_download` or `remove_model`
  and a `model_id`, or `acceleration` / `unload_timeout` and a `value`.
  Acceleration accepts `auto`, `cpu`, or `gpu:<device-id>` for a ggml model.
  ONNX models expose Auto/CPU. Unload values are `never`, `immediately`, `min_2`,
  `min_5`, `min_10`, `min_15`, and `hour_1`.

One inference/model operation runs at a time. Conflicting operations return HTTP
409; status remains available. This first transport supports batch transcription.
Handy's existing idle-unload and accelerator settings apply to its engine.
Download cancellation remains available while downloading. Model selection and
runtime settings persist across restarts. Streaming is not exposed by this API.

A plain Linux Cargo build stages libraries in `src-tauri/transcribe-libs`; set
`LD_LIBRARY_PATH` to that directory when running the development executable.
Packaged installations use their bundled library path.

Audio is validated and held only in memory. The API accepts neither arbitrary
filesystem paths nor model URLs. It rejects Origin-bearing browser requests and
unexpected Host headers and sends no CORS grants. Keep the token private; it is
separate from any T3 pairing or provider credential.
