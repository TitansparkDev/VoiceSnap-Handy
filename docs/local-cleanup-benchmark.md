# Local cleanup candidate benchmark

Use the fixed-fixture cleanup benchmark before choosing a Wave 1 local model. Candidate names are labels only; compare measured correctness and cleanup latency instead of selecting a winner from the model name.

## Requirements

- Node.js available in the development checkout.
- A local OpenAI-compatible cleanup runtime such as `llama-server`.
- Candidate model assets already present on disk.
- The runtime must listen on a loopback endpoint. The harness rejects non-loopback endpoints and performs no model downloads.

The benchmark sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` for the child runtime and bypasses proxies. Its own HTTP transport rejects every non-loopback host. It can therefore be run with external networking disabled when the runtime binary and candidate assets are already local.

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

Use local Q4 assets for Qwen3-1.7B, Qwen3-4B, and S1-mini by Superwhisper. File names below are examples; point them at the actual local files. S1-mini has a trained prompt contract, so select its explicit profile instead of sending the generic cleanup prompt:

```bash
npm run bench:cleanup -- \
  --candidate qwen3-1.7b=/models/Qwen3-1.7B-Q4_K_M.gguf \
  --candidate qwen3-4b=/models/Qwen3-4B-Q4_K_M.gguf \
  --candidate s1-mini=/models/s1-mini-q4_k_m.gguf \
  --candidate-profile s1-mini=s1-mini-v1 \
  --include-configured \
  --json cleanup-benchmark.json
```

Each candidate is loaded independently through the configured local runtime. Stop any existing process using the benchmark endpoint before starting so one already-running model cannot accidentally service another candidate's measurements.

`generic-v1` is the default prompt profile. `s1-mini-v1` uses S1-mini's published system prompt, `[Styling: semi-formal] [Structure: prose] [Context: general]` control line, greedy decoding, and disabled thinking. This is a candidate-specific input protocol, not a different fixture set: all candidates still receive the same five transcript fixtures and the same accepted-output checks.

## What is measured

The fixed fixture set exercises one-sentence cleanup cases for punctuation, number words, spoken punctuation, filler removal, questions, and transcript text that looks like an instruction. The `punctuation` case is the designated short-dictation fixture. The harness runs every fixture three times by default and reports a dedicated repeatable p50/p95/max for that fixture in addition to the aggregate candidate timing. It records:

- candidate label and prompt profile;
- local model asset file name, size, and not the full path;
- request success/failure;
- accepted-output correctness;
- cleanup latency per fixture;
- candidate p50, p95, and maximum cleanup latency;
- designated short-dictation fixture p50, p95, and maximum cleanup latency, plus per-fixture timing summaries;
- runtime startup time;
- observed child-process RSS on Linux (startup and maximum sampled RSS).

Use `--runs N` to change the sample count. `--request-timeout-ms` defaults to 20,000 ms and bounds each cleanup request; `--startup-timeout-ms` defaults to 8,000 ms and bounds runtime readiness.

The JSON result intentionally does **not** contain fixture input text, model output text, transcript history, audio, clipboard contents, window titles, API keys, or full local file paths. It includes an `offline_contract` block recording that external network dependency is disabled, model assets are local-only, model-hub lookups are disabled, and proxy use is bypassed. The committed fixture text is static test data and is only sent to the loopback runtime during the benchmark.

The automated harness test also inspects the exact completion request shape: the local model receives only the cleanup system instruction plus transcript text and generation controls. No audio, clipboard contents, window title, foreground-application identity, or unrelated application data field exists in the request contract. A second test fixes three timing samples for the designated one-sentence dictation and proves the emitted p50/p95 are deterministic and contain no fixture text.

A candidate should only be considered after checking both correctness and latency. A lower p50/p95 does not compensate for failed fixtures, malformed cleanup output, or repeated request failures. Linux RSS is sampled after startup and after each request; it is an observation, not a platform-independent peak-memory guarantee.

## Recorded CPU comparison — 3 September 2026

This repository benchmark was run on Linux 6.8, x86_64, with 16 logical CPUs and 15.5 GiB RAM. It used the llama.cpp b10549 Ubuntu x64 CPU binary (archive SHA-256 `66b26d8cb3ab8edaf5a12bfe642b8f00844925f614f196a96a222b7ed1582c1d`), 8 inference threads, a 2,048-token context, Jinja chat templates, and `-ngl 0` to force CPU fallback. Every candidate used `wave1-cleanup-short-v1`, 5 fixtures × 3 measured requests.

| Candidate | Q4 asset SHA-256 | Prompt profile | Correct | p50 | p95 | Startup | Max sampled RSS | Asset size |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5` | `generic-v1` | 0/15 | 492 ms | 1,341 ms | 4,256 ms | 2,166.5 MiB | 1,282,439,264 B |
| Qwen3-4B | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | `generic-v1` | 3/15 | 1,121 ms | 4,646 ms | 6,505 ms | 4,428.0 MiB | 2,497,280,256 B |
| S1-mini by Superwhisper | `3b41ebe2502cbd03e811d5d16b022f5ab551eda58d62597d152f89535003c634` | `s1-mini-v1` | 15/15 | 189 ms | 505 ms | 2,468 ms | 965.4 MiB | 484,219,808 B |

All 45 requests completed successfully at the transport/runtime level. Qwen3-4B passed only the punctuation fixture across its three runs; the remaining exact cleanup fixtures failed. Qwen3-1.7B produced valid text responses but none matched the accepted cleanup outputs. S1-mini passed all five fixtures on all three runs. On this CPU host it also had the lowest p50/p95, startup time, sampled RSS, and asset size. On measured quality and resource evidence, **S1-mini is the preferred English cleanup candidate for further integration testing**; that preference is not based on its name or published benchmark claims.

Reproduction command after the three assets and runtime are already local:

```bash
npm run bench:cleanup -- \
  --runtime-command /opt/llama-b10549/llama-server \
  --runtime-args-json '["--host","127.0.0.1","--port","8080","--threads","8","--ctx-size","2048","--jinja","-ngl","0"]' \
  --candidate qwen3-1.7b=/models/Qwen3-1.7B-Q4_K_M.gguf \
  --candidate qwen3-4b=/models/Qwen3-4B-Q4_K_M.gguf \
  --candidate s1-mini=/models/s1-mini-q4_k_m.gguf \
  --candidate-profile s1-mini=s1-mini-v1 \
  --runs 3 \
  --request-timeout-ms 60000 \
  --startup-timeout-ms 30000 \
  --json cleanup-cpu.json
```

The benchmark execution itself is local/offline: model paths are local files, the HTTP endpoint is loopback-only, and the harness forces Hugging Face/Transformers offline environment variables for the child runtime. Acquiring the runtime and model assets is a separate setup step.

### Candidate availability and licensing record

- **Qwen3-1.7B Q4_K_M** — evaluated from the published `ggml-org/Qwen3-1.7B-GGUF` asset. The model repository declares Apache-2.0.
- **Qwen3-4B Q4_K_M** — evaluated from the published `Qwen/Qwen3-4B-GGUF` asset. The model repository declares Apache-2.0.
- **S1-mini by Superwhisper Q4_K_M** — evaluated from the published `superwhisper/s1-mini-GGUF` asset. Its published license is Apache-2.0 plus an additional naming/attribution term requiring the model to remain named “S1-mini” by “Superwhisper”; redistribution also requires its license/NOTICE obligations. Treat benchmark suitability and redistribution approval as separate decisions.
- **VoiceTypr source** — not used or copied. VoiceTypr is AGPL-3.0; this benchmark only reimplements the evaluation idea and uses independently published model assets/runtime interfaces.

No listed candidate was silently treated as evaluated: all three assets were legitimately obtainable and were actually measured in this run. If a future candidate asset is absent, inaccessible, or fails license review, record it as unavailable/license-blocked and leave its latency/correctness fields unevaluated rather than substituting another model.

## Command help and harness tests

```bash
npm run bench:cleanup -- --help
node --test scripts/cleanup-benchmark.test.mjs
```
