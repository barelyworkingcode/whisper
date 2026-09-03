# Whisper STT Daemon

A local speech-to-text daemon that keeps a Whisper model loaded in memory with Apple Silicon MLX acceleration. Send base64-encoded audio over TCP and get back transcribed text as JSON.

Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast inference on Apple Silicon. Default model: `mlx-community/whisper-large-v3-turbo`.

## Requirements

- macOS with Apple Silicon
- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Miniforge)
- [Homebrew](https://brew.sh/) (for ffmpeg)

## Setup

```bash
./build.sh
```

This will:
1. Create a `whisper` conda environment with Python 3.11 and dependencies (`mlx-whisper`, `soundfile`, `numpy`)
2. Install `ffmpeg` via Homebrew if not present
3. Register the daemon with [Relay](https://relay.dev) for auto-start (if Relay is installed)

## Usage

### Start the daemon

```bash
./daemon/daemon_wrapper.sh
```

The daemon listens on `localhost:9998` by default. Options:

```bash
./daemon/daemon_wrapper.sh --port 9998 --model mlx-community/whisper-large-v3-turbo --idle-timeout 900
```

The daemon auto-shuts down after 15 minutes of inactivity by default (set `--idle-timeout 0` to disable).

### Transcribe audio

```bash
./test_transcribe.sh audio.wav          # transcribe a file
./test_transcribe.sh audio.wav en       # with language hint
./test_transcribe.sh --ping             # health check
```

### TCP protocol

Clients connect over TCP and exchange length-prefixed JSON messages (4-byte big-endian header).

**Request:**
```json
{
  "audio_base64": "<base64-encoded audio>",
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "text": "transcribed text",
  "language": "en",
  "duration": 3.5,
  "transcription_time": 0.42
}
```

**Health check:**
```json
{"action": "ping"}
```

Audio is automatically converted to 16kHz mono WAV via ffmpeg if needed.

## Remote inference

By default the daemon loads Whisper itself. Give it a `--remote-url` and it loads **nothing** — no mlx-whisper, no weights, no `gen_lock` — and transcribes by calling an OpenAI-compatible `POST {base_url}/audio/transcriptions` instead. Measured footprint: **28 MB** resident.

This is for running the daemon on a machine too small to hold the weights — a VM, a spare box — while something with a GPU does the work. Nothing else changes: the payload guards, the ffmpeg conversion to 16 kHz mono, the duration probe and the TCP protocol on 9998 are all identical, so clients cannot tell which engine is serving them.

```bash
./setup_env.sh --remote     # soundfile + numpy only, no mlx-whisper

WHISPER_REMOTE_URL=http://<router>:<port>/v1 WHISPER_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
```

Or directly:

```bash
python daemon/whisper_daemon.py   --remote-url http://<router>:<port>/v1   --remote-model <id-that-server-exposes>
```

Notes:

- `--remote-model` is the id the **remote** server exposes, which is not `--model`. A router may prefix or alias its upstreams.
- A URL is the whole switch — there is no separate enable flag to drift out of sync with it.
- If a router fronts the inference server, point at the router: it can hold the upstream credential, so no token has to live beside the daemon. `--remote-api-key-env` names the variable to read if the endpoint does authenticate; the token is never a command-line argument, where it would show up in the process list.
- A bad endpoint surfaces per request, not at startup, so the daemon still comes up and answers `ping` while a remote host is booting.

## Round-trip test

Test the full TTS-to-STT pipeline with a [Kokoro TTS](https://github.com/barelyworkingcode/kokoro) daemon:

```bash
./test_roundtrip.sh                          # default phrase
./test_roundtrip.sh "Custom test phrase"     # custom phrase
```

Requires a Kokoro daemon running on port 9997.
