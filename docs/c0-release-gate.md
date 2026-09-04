# C0 automated release gate

Date: 2026-09-04

This record covers the integrated remaining-wave trunk candidate exercised by the C0 automated gate.

## Final trunk verification

The final post-C0 trunk candidate was re-verified on 2026-09-04 after all queued feature work had landed. This pass found and repaired two integrated trunk regressions before continuing: untranslated literal labels in the new Performance settings UI caused `bun run lint` to fail, and the diagnostics percentile helper triggered Rust 1.96's `manual_div_ceil` Clippy lint. After those repairs, the exact release commands passed on the available Linux host:

- `bun run lint`
- `bun run build`
- `cargo fmt --manifest-path src-tauri/Cargo.toml -- --check`
- `cargo clippy --manifest-path src-tauri/Cargo.toml --locked --all-targets -- -D warnings`
- `cargo build --manifest-path src-tauri/Cargo.toml --locked`
- `cargo test --manifest-path src-tauri/Cargo.toml --locked` — 413 Rust unit tests passed, with no failures.

Focused final checks also passed:

- `node --test scripts/cleanup-benchmark.test.mjs` — 13/13, including loopback-only/offline runtime policy, ambient-data exclusion, output non-persistence, candidate prompt contracts, and deterministic percentile coverage.
- `node --test scripts/stream-benchmark.test.mjs` — 13/13, including the Wave 2 model matrix, availability preflight, fixed short/medium/long fixtures, live-microphone command/schema coverage, final-only fallback, timing aggregation, and text-free reference reporting.
- focused Rust suites: diagnostics/performance 5/5, cleanup 32/32, stream 10/10, live insertion 12/12, vocabulary 17/17, history 42/42, GPU fallback 10/10, audio-device recovery 4/4, media 17/17, clipboard 13/13, and paste transaction 11/11.
- `bun run check:model-languages`.
- `npx playwright test tests/app.spec.ts --grep "Recording overlay"`.

The Linux run executes the cross-platform clipboard transaction state-machine coverage, including newer-owner-wins and delayed-render/clipboard-manager race stress. The Windows-only STGMEDIUM materialization and timing-budget tests remain present in `src-tauri/src/paste_tx/windows.rs` and are documented in `docs/windows-clipboard-audit.md`; this Linux verification does not falsely claim to have executed Windows-only test binaries.

There is no separately enabled stop-time no-speech product gate in this final trunk. The automated safety evidence is the positive speech-evidence latch and VAD behavior: live committed insertion downgrades to preview without positive speech evidence, committed-looking hallucinations wait for the latch, and silence is rejected by the Earshot VAD test. The PLAN quiet-speech/manual no-speech matrix therefore remains manual rather than being claimed by this automated run.

Packaged-build diagnostics visibility and the cross-platform native application smoke matrix also remain manual gates. Automated PLAN items are only considered complete where the current code/tests above provide direct evidence.

## Automated checks

Passed on the available Linux host:

- `bun run lint`
- `bun run build`
- `cargo fmt --manifest-path src-tauri/Cargo.toml -- --check`
- `cargo clippy --manifest-path src-tauri/Cargo.toml --locked --all-targets -- -D warnings`
- `cargo test --manifest-path src-tauri/Cargo.toml --locked` (413 tests; no failures)
- `cargo build --manifest-path src-tauri/Cargo.toml --locked`
- focused cleanup, output/final-paste, coordinator/live-insertion, streaming/hardware, vocabulary, history, media, audio-device and clipboard suites
- `node --test scripts/cleanup-benchmark.test.mjs` (13/13)
- `node --test scripts/stream-benchmark.test.mjs` (13/13)
- `bun run check:model-languages`
- `npx playwright test tests/app.spec.ts --grep "Recording overlay"` (1/1)

Focused privacy assertions pass for release-log redaction, stream timing serialization, cleanup request schema, performance export schema and categorical media diagnostics. The authoritative landing verification reruns the repository-wide gate after rebasing onto the latest trunk.

## PLAN reconciliation boundaries

The 2026-09-04 C0 pass reconciles AI-executable PLAN items only where current code and automated evidence directly satisfy them. The following items intentionally remain unclaimed rather than being converted into automated passes:

- packaged-build Diagnostics visibility and real-dictation UI behavior, which require installing/running a packaged application;
- the Wave 2 real-model timing/quality matrix and live-model acceptance, because the audited host has none of the four Wave 2 model assets installed and the repository has no valid short/medium/long 16 kHz mono speech/reference fixture set;
- the Wave 8 Windows-only STGMEDIUM/timing binary tests on this Linux host, plus Word/Office/browser/Electron/Qt/elevated-window smoke;
- real-device/no-GPU/NVIDIA/AMD/Intel, microphone-disappearance, media-player and cross-platform application smoke;
- the cleanup-disabled versus historical Handy baseline timing comparison, which needs equivalent real dictation runs rather than a code-path assertion;
- quiet-speech/no-speech manual validation. The automated safety claim is limited to the positive speech-evidence latch, conservative preview downgrade and VAD tests.
- Wave 4's optional snippets clause was not selected for this track. The vocabulary contract, aliases and deterministic replacements are complete without adding a snippet feature that could alter the streaming insertion contract.

These are environment/manual/conditional validation boundaries, not hidden repository implementation work. No model timing, native application result or platform-specific test is inferred from catalog metadata or from a Linux-only run.

## Privacy assertions

The automated checks establish that:

- release builds replace transcript-bearing diagnostic log text with `[REDACTED]`; debug-only raw preview remains an explicit development behavior;
- serialized stream performance timing contains no transcript, audio, clipboard or window-title fields;
- cleanup request schemas contain transcript text plus model/control fields only and do not attach audio, clipboard, window-title or ambient application data;
- the offline cleanup benchmark makes the same ambient-data exclusion assertion and does not persist output text;
- media failure diagnostics expose categorical status instead of backend detail.

No diagnostics export path in the current repository serializes raw transcript, audio, clipboard or window-title content by default.

## Model catalog, bundle and attribution review

`AppSettings::default()` has no selected ASR model. Catalog ASR weights are fetched at runtime rather than embedded in the application resources. Local cleanup weights are supplied through a local runtime/model path rather than bundled by the application.

The application resources contain two model-related assets:

- `resources/models/silero_vad_v4.onnx` — Silero VAD, MIT;
- `resources/models/gigaam_vocab.txt` — GigaAM vocabulary, MIT.

`resources/THIRD_PARTY_MODEL_NOTICES.md` now ships the source, copyright notices and MIT terms for those bundled assets. Catalog entries with attribution, non-commercial or otherwise restricted licenses remain runtime-download choices; this release gate does not convert them into bundled redistribution. A future change that embeds catalog model weights must audit that model's individual license before shipping it.

No unresolved redistribution blocker was found for the model-related assets currently bundled by the package.

## Packaged-build host limitation

The W execution bridge refused `bun run tauri build --debug` and `bun run tauri build --debug --no-bundle` before command execution because this queued job is not marked as artifact-producing. This is an execution-policy gate, not a compiler/linker/package failure from the repository, so there was no repository-fixable packaged-build error to repair. The available host did successfully complete the production frontend build and locked Rust application build.

## Final-paste defaults

The focused output and coordinator suites confirm that insertion defaults remain `AtStop`, live insertion remains behind the experimental master switch, non-streaming sessions fall back to at-stop behavior, and live delivery/safety stops prevent duplicate whole-transcript repaste.
