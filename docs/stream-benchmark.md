# Streaming benchmark

Wave 2 uses the real headless transcription path to compare live-capable models and final-only fallback with both deterministic fixed WAV fixtures and real microphone sessions. Both input modes use the same timing/quality schema; the benchmark does not infer speed from catalog labels or model names.

## Build and fixtures

Build Handy first, then prepare fixed 16 kHz mono 16-bit PCM WAV files. Keep the same files for every candidate. The harness records each fixture's SHA-256 so results from different machines/runs can be checked against the same audio without exporting transcript content.

A useful fixture set contains short (under 2 seconds), medium (5–15 seconds), and long (30–120 seconds) speech. The WAV files may live outside the repository; pass stable labels such as `short`, `medium`, and `long`. For repeatable quality measurement, keep a UTF-8 reference transcript beside each fixture and pass it with the same label. Handy computes word-error rate (WER) locally and emits only the numeric rate plus the reference file basename/SHA-256; expected and recognized text are not stored in benchmark output.

## One model directly

The Handy binary has a timing-only headless mode:

```bash
/path/to/handy \
  --transcribe-file /bench/normal.wav \
  --model handy-computer/parakeet-unified-en-0.6b-gguf \
  --benchmark-stream \
  --benchmark-reference-file /bench/normal.txt \
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

Benchmark JSON contains timing, worker-release state, safe model/backend metadata, and optional `word_error_rate_milli` (0 = exact match, 1000 = 100% WER). It does **not** contain transcript/reference text, raw audio, clipboard contents, window titles, document text, or full fixture paths.

## Live microphone sessions

Use `--benchmark-live-seconds N` to open the selected microphone through `AudioRecordingManager`, disable VAD filtering for the benchmark capture, and feed real admitted frames into the same `StreamRouter` used by normal live dictation. At stop, streaming engines finalize the active stream; engines without a usable streaming path fall back to the same batch transcription route used by the fixed-WAV benchmark.

```bash
/path/to/handy \
  --model handy-computer/parakeet-unified-en-0.6b-gguf \
  --benchmark-live-seconds 10 \
  --benchmark-reference-file /bench/live-phrase.txt \
  --repeat 3 \
  --json
```

Microphone readiness is bounded: if the device opens but supplies no samples within five seconds, the benchmark cancels the recorder and stream instead of hanging. Actual live measurements necessarily require a microphone, but automated Node tests verify argument parsing, child-command construction, catalog routing, timing/quality aggregation, final-only behavior, and privacy without touching physical hardware.

## Compare the Wave 2 catalog candidates

Use the aggregate harness to run identical fixed WAVs through the current Wave 2 matrix:

```bash
npm run bench:stream -- \
  --binary /path/to/handy \
  --fixture short=/bench/short.wav \
  --reference short=/bench/short.txt \
  --fixture medium=/bench/medium.wav \
  --reference medium=/bench/medium.txt \
  --fixture long=/bench/long.wav \
  --reference long=/bench/long.txt \
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

The aggregate report records p50/p95 for first partial, committed cadence, finalization tail, total time, and optional WER. It also records load time, bound backend, fixture/reference basename and SHA-256, catalog-path classification, success/failure, and worker-release state. `--wave2-models` is checked against the live catalog in automated tests: Parakeet, Nemotron, and Moonshine are catalog streaming candidates, while Whisper Large v3 Turbo is explicitly final-only. A result is not successful if the worker/lease remains active after finalization.

The aggregate harness can also run a live session for each selected candidate in the same invocation:

```bash
npm run bench:stream -- \
  --binary /path/to/handy \
  --live-seconds 10 \
  --live-label live-phrase \
  --live-reference /bench/live-phrase.txt \
  --wave2-models \
  --runs 3 \
  --json live-stream-benchmark.json
```

## Offline behavior

The benchmark does not download models. Every requested model must already be installed locally. The aggregate runner also starts child processes with Hugging Face/Transformers offline environment flags. It therefore remains usable with networking disabled when the Handy binary, models, and fixed WAV fixtures are already local.

## Focused tests

```bash
node --test scripts/stream-benchmark.test.mjs
cargo test --manifest-path src-tauri/Cargo.toml benchmark_word_error_rate_is_deterministic_and_text_free
cargo test --manifest-path src-tauri/Cargo.toml overlay_activation_policy_never_requests_target_focus
cargo test --manifest-path src-tauri/Cargo.toml cancellation_quiesces_worker_and_allows_next_session_reservation
cargo test --manifest-path src-tauri/Cargo.toml model_switch_quiesce_releases_old_lease_before_new_worker_generation
cargo test --manifest-path src-tauri/Cargo.toml stale_stream_worker_cannot_clear_new_model_switch_worker
cargo test --manifest-path src-tauri/Cargo.toml stream_text_event_serializes_committed_and_tentative_as_distinct_fields
cargo test --manifest-path src-tauri/Cargo.toml stream_perf_records_timing_only_and_committed_cadence
npx playwright test tests/app.spec.ts --grep "Recording overlay"
```

The lifecycle tests cover cancellation cleanup and generation fencing across model/session switches. The browser test proves committed text is visually stronger than tentative text. The native overlay-policy test asserts the window is non-focusable, not initially focused, and never intentionally activated when shown; production platform code retains Windows `SWP_NOACTIVATE`, macOS `no_activate`/nonactivating-panel behavior, and Linux layer-shell `KeyboardMode::None`.
