# C0 automated release gate

Date: 2026-09-03

This record covers the integrated remaining-wave trunk candidate exercised by the C0 automated gate.

## Automated checks

Passed on the available Linux host:

- `bun run lint`
- `bun run build`
- `cargo fmt --manifest-path src-tauri/Cargo.toml -- --check`
- `cargo clippy --manifest-path src-tauri/Cargo.toml --locked --all-targets -- -D warnings`
- `cargo test --manifest-path src-tauri/Cargo.toml --locked` (397 tests before the C0 privacy assertion was added; no failures)
- `cargo build --manifest-path src-tauri/Cargo.toml --locked`
- focused cleanup, output/final-paste, coordinator/live-insertion, streaming/hardware, history, media, audio-device and clipboard suites
- `node --test scripts/cleanup-benchmark.test.mjs`

After the C0 fixes, focused privacy assertions also pass for release-log redaction, stream timing serialization, cleanup request schema and categorical media diagnostics. The authoritative landing verification is expected to rerun the repository-wide test gate with the added privacy assertion.

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
