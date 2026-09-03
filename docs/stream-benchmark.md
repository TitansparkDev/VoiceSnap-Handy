# Fixed-WAV streaming benchmark

Wave 2 uses the real headless transcription path to compare live-capable models and final-only fallback on identical local WAV fixtures. The benchmark does not infer speed from catalog labels or model names.

## Build and fixtures

Build Handy first, then prepare fixed 16 kHz mono 16-bit PCM WAV files. Keep the same files for every candidate. The harness records each fixture's SHA-256 so results from different machines/runs can be checked against the same audio without exporting transcript content.

A useful fixture set contains short (under 2 seconds), medium (5–15 seconds), and long (30–120 seconds) speech. The WAV files may live outside the repository; pass stable labels such as `short`, `medium`, and `long`.

## One model directly

The Handy binary has a timing-only headless mode:

```bash
/path/to/handy \
  --transcribe-file /bench/normal.wav \
  --model handy-computer/parakeet-unified-en-0.6b-gguf \
  --benchmark-stream \
  --repeat 3 \
  --json
```

The WAV is fed through `StreamRouter` in real-time-sized frames. Streaming-capable engines report:

- `first_partial_ms`: start of the stream request to the first emitted live text update;
- `committed_cadence_ms`: intervals between committed-text changes;
- `finalization_tail_ms`: stop/finalize request to final transcription return;
- `total_ms`: stream request to final transcription return;
- `worker_released`: whether the streaming worker and engine lease fully released within the bounded post-run check.

For a model that cannot stream, Handy intentionally falls back to the normal batch transcription path. Its result uses `mode: "final_only"`, leaves `first_partial_ms` null, leaves `committed_cadence_ms` empty, and reports the batch decode as the finalization tail. The harness never treats missing partials as a failure for a final-only model.

Benchmark JSON contains timing and safe model/backend metadata plus output character count. It does **not** contain transcript text, raw audio, clipboard contents, window titles, document text, or full fixture paths.

## Compare the Wave 2 catalog candidates

Use the aggregate harness to run identical fixed WAVs through the current Wave 2 matrix:

```bash
npm run bench:stream -- \
  --binary /path/to/handy \
  --fixture short=/bench/short.wav \
  --fixture medium=/bench/medium.wav \
  --fixture long=/bench/long.wav \
  --wave2-models \
  --runs 3 \
  --json stream-benchmark.json
```

`--wave2-models` expands to the current catalog IDs for:

- Parakeet Unified EN 0.6B;
- Nemotron Streaming 3.5;
- Moonshine Streaming Tiny;
- Whisper Large v3 Turbo.

You can instead repeat `--model MODEL_ID` for any installed catalog/custom model. Use `handy --list-models` to inspect local IDs and `handy --list-devices` plus `--device-index N` when an exact transcribe-cpp device must be compared.

The aggregate report records p50/p95 for first partial, committed cadence, finalization tail, and total time. It also records load time, bound backend, fixture basename/SHA-256, success/failure, and worker-release state. A result is not successful if the worker/lease remains active after finalization.

## Offline behavior

The benchmark does not download models. Every requested model must already be installed locally. The aggregate runner also starts child processes with Hugging Face/Transformers offline environment flags. It therefore remains usable with networking disabled when the Handy binary, models, and fixed WAV fixtures are already local.

## Focused tests

```bash
node --test scripts/stream-benchmark.test.mjs
cargo test --manifest-path src-tauri/Cargo.toml cancellation_quiesces_worker_and_allows_next_session_reservation
cargo test --manifest-path src-tauri/Cargo.toml model_switch_quiesce_releases_old_lease_before_new_worker_generation
cargo test --manifest-path src-tauri/Cargo.toml stale_stream_worker_cannot_clear_new_model_switch_worker
cargo test --manifest-path src-tauri/Cargo.toml stream_text_event_serializes_committed_and_tentative_as_distinct_fields
cargo test --manifest-path src-tauri/Cargo.toml stream_perf_records_timing_only_and_committed_cadence
```

The lifecycle tests cover cancellation cleanup and generation fencing across model/session switches. They fail if an old worker can clear a newer lease or if a current worker guard leaves active state behind.
