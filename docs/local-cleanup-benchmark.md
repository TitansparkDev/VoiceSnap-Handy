# Local cleanup candidate benchmark

Use the fixed-fixture cleanup benchmark before choosing a Wave 1 local model. Candidate names are labels only; compare measured correctness and cleanup latency instead of selecting a winner from the model name.

## Requirements

- Node.js available in the development checkout.
- A local OpenAI-compatible cleanup runtime such as `llama-server`.
- Candidate model assets already present on disk.
- The runtime must listen on a loopback endpoint. The harness rejects non-loopback endpoints and performs no model downloads.

The benchmark also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the child runtime. It can therefore be run with networking disabled when the runtime binary and candidate assets are already local.

## Runtime configuration

The harness understands the same environment variables used by VoiceSnap-Handy's supervised local cleanup runtime:

```text
HANDY_LOCAL_CLEANUP_COMMAND=/path/to/llama-server
HANDY_LOCAL_CLEANUP_ARGS=["--host","127.0.0.1","--port","8080","-m","/models/current.gguf","-ngl","99"]
```

`HANDY_LOCAL_CLEANUP_ARGS` must be a JSON array. For each candidate, the harness keeps the configured runtime arguments and replaces the model passed through `-m`, `--model`, or `--model-file`. If none is present it appends `-m MODEL_PATH` by default; use `--model-flag` for another compatible runtime.

With no explicit `--candidate`, the harness benchmarks the current model path found in `HANDY_LOCAL_CLEANUP_ARGS`:

```bash
npm run bench:cleanup
```

Add `--include-configured` when comparing the current configured model against other local assets.

## Compare Wave 1 candidates

Use local Q4 assets for Qwen3-1.7B, Qwen3-4B, and an s1-mini-compatible model. File names below are examples; point them at the actual local files:

```bash
npm run bench:cleanup -- \
  --candidate qwen3-1.7b=/models/Qwen3-1.7B-Q4_K_M.gguf \
  --candidate qwen3-4b=/models/Qwen3-4B-Q4_K_M.gguf \
  --candidate s1-mini=/models/s1-mini-Q4_K_M.gguf \
  --include-configured \
  --json cleanup-benchmark.json
```

Each candidate is loaded independently through the configured local runtime. Stop any existing process using the benchmark endpoint before starting so one already-running model cannot accidentally service another candidate's measurements.

## What is measured

The fixed fixture set exercises short cleanup cases for punctuation, number words, spoken punctuation, filler removal, questions, and transcript text that looks like an instruction. The harness runs every fixture three times by default and records:

- candidate label;
- local model asset file name, not the full path;
- request success/failure;
- accepted-output correctness;
- cleanup latency per fixture;
- candidate p50, p95, and maximum cleanup latency.

Use `--runs N` to change the sample count. `--request-timeout-ms` defaults to 20,000 ms and bounds each cleanup request; `--startup-timeout-ms` defaults to 8,000 ms and bounds runtime readiness.

The JSON result intentionally does **not** contain fixture input text, model output text, transcript history, audio, clipboard contents, window titles, API keys, or full local file paths. The committed fixture text is static test data and is only sent to the loopback runtime during the benchmark.

A candidate should only be considered after checking both correctness and latency. A lower p50/p95 does not compensate for failed fixtures, malformed cleanup output, or repeated request failures.

## Command help and harness tests

```bash
npm run bench:cleanup -- --help
node --test scripts/cleanup-benchmark.test.mjs
```
