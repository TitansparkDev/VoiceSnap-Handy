# VoiceSnap-Handy plan

This is the implementation plan for the VoiceSnap fork of Handy.

Repository: `TitansparkDev/VoiceSnap-Handy`
Upstream: `cjpais/Handy`
Base: upstream `main` at the time this plan was created

The purpose of this fork is to keep Handy's very fast, in-process transcription
core while adding the most valuable VoiceSnap and VoiceTypr capabilities:

- genuinely useful live transcription;
- optional committed-text live insertion;
- local offline AI cleanup;
- visible latency diagnostics;
- better vocabulary handling;
- richer searchable history;
- reliable hardware and device fallback;
- optional media pause;
- conservative Windows clipboard hardening.

This plan is intentionally narrower than either reference project. It does not
include LAN sharing, file transcription, cloud STT, mouse side-button shortcuts,
an onboarding wizard, or app-aware formatting profiles in the first track.

## Decisions already made

These are user decisions, not open questions:

1. Use Handy as the foundation because its resident in-process streaming path is
   already closer to the desired latency than VoiceSnap's HTTP sidecar path.
2. Add the live overlay/preview experience.
3. Add an opt-in experimental mode that inserts stable committed text while the
   user speaks.
4. Add a no-speech gate, but expose it as a setting. It may begin disabled or
   shadow-only until the false-negative tests pass.
5. Add real, visible latency diagnostics. Hidden logs are not sufficient.
6. Do not add app-aware writing profiles in this track.
7. Build a better vocabulary system, preserving Handy's current custom-word
   behavior and improving it with spoken aliases and language scope.
8. Add richer history.
9. Keep Handy's existing paste mechanism unless VoiceSnap's implementation has
   a demonstrable correctness improvement. Merge only tested improvements.
10. Add optional media pause/resume.
11. Keep everything local/offline by default. Cloud support is not part of the
    core fork plan.

## Non-negotiable product invariants

- The app must remain useful with no GPU and on Windows 10/11, macOS, and Linux
  where Handy currently supports them.
- No Python, Ollama, CUDA-only path, or ROCm-only path is introduced.
- Capture, VAD, resampling, clipboard, history, and UI work must not block the
  audio callback or the streaming inference worker.
- A model may emit tentative text, but tentative text must never be inserted
  into the user's target application.
- The default insertion mode remains one final paste at stop.
- Experimental live insertion is opt-in and visibly labelled experimental.
- If live insertion is enabled, the app must never re-paste the whole growing
  transcript on every update.
- Live insertion must not begin until the recording has crossed a speech-evidence
  latch. A final no-speech decision must never leave already-inserted hallucinated
  text behind.
- Live insertion uses one explicitly defined text-transform contract. It must not
  insert raw stream deltas and then silently produce a different cleaned final
  transcript.
- AI cleanup must fail open: if local cleanup fails, times out, or returns an
  invalid result, preserve and optionally paste the raw transcript.
- User clipboard contents win. A paste must not overwrite a newer clipboard
  owner or newer user copy.
- Focus changes, cancellation, engine failure, and model unload must not strand
  an active stream or cause text to be inserted into the wrong window.
- No raw transcript, audio, clipboard contents, window title, or document text
  is written to diagnostic telemetry by default.
- Performance samples live in a bounded diagnostics store, not the transcription
  history table. History may reference a session ID and safe timing summaries,
  but timing collection must not create a second transcript persistence path.
- Every change must add focused tests and a manual smoke item when it affects
  native input, audio, model lifecycle, or clipboard behavior.

## Current Handy baseline to preserve

Handy already contains several pieces that should not be rebuilt:

- in-process `transcribe-cpp` and `transcribe-rs` engines;
- a `StreamRouter` that feeds live 16 kHz frames into a streaming session;
- separate committed and tentative stream text events;
- model capability detection and a data-driven catalog;
- VAD prefill and streaming hangover behavior;
- stateful resampling and a real-time-safe audio callback;
- warm/always-on recording support;
- model preloading, unload timeouts, cancellation, and stream leases;
- custom words and deterministic transcription text cleanup;
- SQLite-backed history with raw and post-processed text;
- receipt-sequenced clipboard paste, currently treated as a beta/debug path;
- platform-specific clipboard and input implementations;
- global shortcut and hold/toggle coordination;
- optional provider-based post-processing;
- model hashes, revisions, quantization choices, and custom model discovery.

Relevant upstream source:

- [transcription manager](https://github.com/cjpais/Handy/blob/main/src-tauri/src/managers/transcription.rs)
- [recording actions](https://github.com/cjpais/Handy/blob/main/src-tauri/src/actions.rs)
- [audio recorder](https://github.com/cjpais/Handy/blob/main/src-tauri/src/audio_toolkit/audio/recorder.rs)
- [model catalog](https://github.com/cjpais/Handy/blob/main/src-tauri/src/catalog/catalog.json)
- [Windows paste transaction](https://github.com/cjpais/Handy/blob/main/src-tauri/src/paste_tx/windows.rs)

## Target user-visible modes

The settings UI should make these modes explicit:

### Final paste

The default and safest mode:

1. Record.
2. Stream to the overlay if the selected model supports streaming.
3. Finalize the model at stop.
4. Optionally run deterministic cleanup and/or local AI cleanup.
5. Paste one final result.

AI cleanup belongs here because it can change earlier words. This mode should
remain the recommended mode for Word, browsers, terminals, and code editors.

### Live preview

The overlay shows:

- stable committed text;
- provisional tentative text;
- microphone state and elapsed time;
- current stage: listening, finalizing, cleaning, or inserting;
- first-partial and last-update timing in diagnostics only.

No text is inserted into the target application until finalization.

### Experimental live insertion

This mode inserts only new committed text while recording. It must:

- maintain a session insertion ledger;
- insert only the delta after the last committed prefix;
- never insert tentative text;
- stop inserting if the foreground window changes;
- avoid modifying the user's clipboard permanently;
- flush the final uncommitted tail once;
- surface a warning that model revisions and application AutoCorrect cannot be
  perfectly reversed in a generic text field.

Live insertion and AI rewriting are incompatible in the general case. The first
implementation must choose one of these safe contracts:

- live insertion uses raw/deterministic text and AI cleanup is disabled for that
  session; or
- live insertion is preview-only when AI cleanup is selected, with one polished
  paste at stop.

Do not silently insert raw text and then attempt to replace it with polished
text. A future app-specific accessibility adapter may support that, but generic
keyboard injection must not.

Before the first live delta is inserted, the audio path must report a positive
speech-evidence latch (or the model must provide equivalent positive speech
confidence). If the recording never reaches that latch, live insertion remains
preview-only for that session. A stop-time no-speech gate cannot undo text that
was already inserted.

## Model strategy

The current Handy catalog provides the initial test matrix:

- **Parakeet Unified EN 0.6B**: first live-English candidate.
- **Nemotron Streaming 3.5 0.6B**: first live multilingual candidate; audit its
  model license before redistribution.
- **Moonshine Streaming Tiny**: low-resource latency fallback, with an expected
  quality tradeoff.
- **Whisper Large v3 Turbo**: accurate final model; catalog capability is
  non-streaming, so do not promise token-level live output.
- **Voxtral Realtime**: defer. It is streaming-capable in the catalog but is not
  currently Handy's recommended speed choice.

Catalog speed/accuracy scores are ranking hints, not measurements for every
machine. The fork needs a repeatable local benchmark before choosing defaults.
Every redistributed model must have an explicit license and attribution audit.

### CPU/GPU policy

The normal runtime lanes are:

- CPU: capture, resampling, VAD, metrics, clipboard, history, and UI plumbing.
- Selected accelerator: one ASR session.
- Cleanup lane: local AI cleanup on the selected accelerator when memory allows,
  otherwise CPU or disabled according to the user's setting.

Do not run two full ASR decodes for every utterance. A later experiment may run a
small streaming model plus Whisper Turbo finalization, but only behind a
benchmark flag and only if it does not cause VRAM contention or make the final
result slower.

## Work waves

### Wave 0 — baseline and visible diagnostics

Owner: one agent only. Other work should consume this contract, not invent
parallel timing formats.

Deliverables:

- Define a monotonic `PerformanceSample` schema.
- Instrument one session with stable stage names:
  - `hotkey_to_capture_ready`
  - `capture_duration`
  - `capture_stop_to_stream_finalize`
  - `capture_stop_to_batch_transcription`
  - `first_partial`
  - `last_partial`
  - `transcription_total`
  - `cleanup_total`
  - `paste_total`
  - `stop_to_visible_text`
  - `stop_to_idle`
  - `total_hotkey_to_idle`
- Record model ID, engine family, language, CPU/GPU selection, actual device,
  recording length, cleanup mode, insertion mode, and success/failure outcome.
- Store a bounded local ring buffer of recent samples. Do not store transcript,
  audio, clipboard, process path, or window title.
- Add a Settings → Diagnostics/Performance page that shows:
  - the latest session as a stage bar;
  - the slowest stage highlighted;
  - p50/p95 for the last 10, 50, and 200 sessions;
  - separate warm and cold-start indicators;
  - model/device/backend labels;
  - first-partial latency for streaming models;
  - a copy/export diagnostics button containing only safe metadata;
  - a clear history button.
- Add a structured debug-log viewer or filtered log export for engine lifecycle,
  stream state, audio recovery, cleanup, and paste outcomes.
- Add tests for timing schema serialization, bounded retention, no transcript
  leakage, and percentile calculations.
- Keep the performance ring separate from SQLite history. If the UI needs a
  history link, store only a non-sensitive session identifier and safe summary
  fields in the history row.

Acceptance:

- After a real dictation, the user can immediately see which stage consumed the
  time without opening a developer console.
- A failed or cancelled session still records a useful stage outcome.
- The diagnostics page makes it possible to compare Whisper Turbo against a
  streaming model on the same machine.

Suggested files: new `performance.rs` or `managers/performance.rs`, settings
command, diagnostics UI, and the existing action/transcription boundaries.

### Wave 1 — local offline AI cleanup

Owner: one backend agent plus one UI agent. Backend and UI can work in separate
branches if the command/event contract is written first.

Deliverables:

- Add a local cleanup provider to Handy's existing post-process abstraction.
- Prefer a resident local `llama.cpp`/`llama-server` process or equivalent
  bundled local runtime rather than launching a new process per utterance. If a
  sidecar is used, it needs a dedicated supervisor with child/job lifetime,
  health checks, bounded startup, cancellation, crash cleanup, and CPU fallback;
  it must not be an unmanaged process launched from the action handler.
- Use a small Q4 model first. Evaluate Qwen3-1.7B, Qwen3-4B, and the VoiceTypr
  `s1-mini` candidate; do not select by model name alone.
- Keep model loading separate from the ASR stream worker.
- Disable reasoning/thinking and constrain output length.
- Define a strict cleanup contract: output text only, no explanation, no quotes,
  no markdown wrapper, no invented content.
- Add cleanup modes:
  - `off`: raw deterministic transcription output;
  - `fast`: code-only corrections and deterministic replacements;
  - `local_ai`: local model cleanup;
  - cloud providers remain optional and are not part of the default path.
- Preserve `raw_text`, `cleaned_text`, cleanup mode, prompt ID, model ID, and
  cleanup timing in history.
- Fail open to raw text on timeout, malformed output, engine failure, or
  cancellation.
- Warm the local model only when local AI cleanup is enabled. Do not load it on
  installations using `off` or `fast` mode.

Acceptance:

- A normal dictation with cleanup disabled is not slower than the Handy baseline.
- Local AI cleanup works with networking disabled.
- Cleanup failure never loses the raw transcript.
- A 1–2 sentence dictation has a measured cleanup p50/p95 visible in Diagnostics.
- The local model never receives audio, clipboard content, window titles, or
  unrelated application data.

### Wave 2 — live preview and stream benchmark

Owner: streaming agent.

Deliverables:

- Preserve Handy's existing streaming worker and committed/tentative event
  contract.
- Add explicit stream performance events to Wave 0's timing schema.
- Add a benchmark command or test harness using fixed WAV fixtures and live
  microphone sessions.
- Compare:
  - Parakeet Unified;
  - Nemotron Streaming;
  - Moonshine Streaming Tiny;
  - Whisper Large v3 Turbo batch;
  - any other model already present in Handy's catalog.
- Measure first partial, committed text cadence, finalization tail, total time,
  and quality on short/medium/long utterances.
- Make the overlay's committed/tentative distinction visually obvious.
- Keep the overlay from stealing focus from the target application.
- Ensure a non-streaming model cleanly falls back to final batch transcription.

Acceptance:

- Live-capable models show text before the user stops speaking.
- Whisper Turbo remains reliable as a final-only model.
- Stream cancellation and model switching do not leak a worker or wedge the next
  session.
- The app reports actual timing instead of claiming a fixed speed.

### Wave 3 — experimental committed-only live insertion

Owner: native input agent. This is the highest-risk feature and must not be
started until Wave 0 and Wave 2 are complete.

Deliverables:

- Add `AtStop`, `PreviewOnly`, and `LiveCommittedExperimental` insertion modes.
- Capture the target window/focus identity at session start.
- Maintain a per-session ledger:
  - committed text already inserted;
  - pending text;
  - insertion attempts and results;
  - focus/ownership changes;
  - cancellation state.
- Insert only the newly committed delta.
- Never paste tentative text.
- Stop live insertion on foreground-window change, target loss, failed input,
  or clipboard ownership loss.
- Finalize and insert the remaining tail exactly once.
- Make the setting and warning explicit. Do not silently activate this mode.
- Keep AI cleanup disabled for live insertion sessions unless the behavior is
  preview-only.
- Require a positive speech-evidence latch before the first live insertion. If
  the latch is absent, do not insert any live text even if the streaming model
  emitted a committed-looking hallucination.
- Do not apply a whole-transcript cleanup transform after live deltas have been
  inserted. Live mode must use either raw text or a separately specified,
  append-safe deterministic transform; AI cleanup remains preview-only or is
  disabled for that session.

Acceptance:

- No duplicate committed text across repeated stream events.
- No full-transcript re-paste loop.
- No insertion into a newly focused application.
- No user clipboard clobber after a normal session or cancellation.
- Word, Notepad, browser text fields, terminal, and a Chromium/Electron field
  are manually tested.
- Any revision of already inserted model text is recorded as a known limitation;
  the app never attempts unsafe generic deletion/replacement.

Stop condition: if a partial is inserted and then revised incorrectly, disable
the feature behind its setting and return the default to final paste.

### Wave 4 — vocabulary and deterministic writing tools

Owner: text-processing agent.

Deliverables:

- Preserve Handy's current `custom_words: Vec<String>` through migration.
- Do not change that field's serialized type in place. Handy's settings salvage
  path can discard an invalid field before a normal serde migration sees it.
  Introduce a new versioned vocabulary key, or migrate the raw JSON value before
  typed deserialization, with tests proving old settings retain every word.
- Extend entries to support:
  - written form;
  - spoken alias;
  - optional language scope;
  - enabled state;
  - optional case/punctuation policy.
- Feed written vocabulary to model paths that support initial prompts or native
  vocabulary. Streaming models that do not accept prompts must use safe
  post-correction only.
- Improve fuzzy correction with bounded, deterministic matching. Never replace
  a word solely because it is vaguely similar.
- Add optional deterministic replacements (`from` → `to`) with language scope.
- Add optional snippets only if they can be implemented without changing the
  streaming insertion contract.
- Sanitize control characters and bound all vocabulary context before it reaches
  a model.
- Add adversarial tests for short words, CJK text, punctuation, aliases,
  disabled entries, duplicate entries, and overmatching.

Acceptance:

- Existing users do not lose their current custom words.
- A spoken alias fixes a known product/name pronunciation without changing
  unrelated words.
- Vocabulary corrections are visible in history as deterministic operations.
- Vocabulary failures fail open to the original transcript.

### Wave 5 — richer history

Owner: history/UI agent. Depends on Wave 0's timing contract and Wave 1's raw/final
cleanup fields.

Deliverables:

- Extend history metadata with model, engine, language, duration, cleanup mode,
  insertion mode, device/backend, stage timings, and outcome.
- Keep raw and final text side by side.
- Add search, saved/starred entries, date filters, model filters, cleanup filters,
  and success/failure filters.
- Add compare raw versus final.
- Add retry transcription and retry cleanup independently.
- Keep audio retention optional and bounded.
- Keep app identity coarse/opt-in if ever added later; do not persist window title
  or document name in this wave.

Acceptance:

- A user can answer “why was this dictation slow?” from its history row.
- Re-cleaning does not require re-recording.
- Deleting history removes associated retained audio according to the selected
  retention policy.

### Wave 6 — hardware, device, and runtime truth

Owner: runtime/hardware agent. Depends on Wave 0's runtime metadata contract.

Deliverables:

- Audit Handy's accelerator detection against actual Vulkan devices, not vendor
  name guesses.
- Prefer a usable discrete device over an integrated/shared-memory device.
- Record saved preference, recommended plan, and actual runtime backend/device
  separately.
- If GPU startup fails or becomes unhealthy, downgrade for the current run only;
  do not rewrite the user's saved preference.
- Keep CPU fallback healthy on systems with no Vulkan runtime.
- Add selected-device diagnostics and a user-facing recovery explanation.
- Audit selected microphone persistence and recovery. An automatic stream repair
  must retry the requested device rather than silently selecting another default.
- Add stable device identity where the platform provides it, while retaining a
  readable device name for the UI.
- Add tests for GPU failure, integrated-vs-discrete selection, no-GPU startup,
  device disappearance, recovery, and preference persistence.

Acceptance:

- GPU fallback is visible and truthful.
- CPU fallback works without changing saved settings.
- A microphone recovery does not silently move dictation to a different mic.
- The diagnostics page shows the actual backend used for each session.

### Wave 7 — optional media pause/resume

Owner: media agent. Independent of the transcription engine.

Deliverables:

- Add a setting, default off.
- On recording start, pause only if media is currently playing.
- Resume only if this session paused it.
- If the user manually pauses/stops media during dictation, do not resume it.
- Use a session/generation ledger so overlapping/cancelled sessions cannot resume
  the wrong state.
- Run platform media calls off the hotkey/start path. A media service timeout or
  failure must never delay microphone capture or first partial text.
- Keep failures non-fatal: transcription continues if media control is absent.
- Show media action/failure in diagnostics without recording media identity.

Acceptance:

- Playback is paused during recording when enabled.
- Playback resumes after successful stop and after cancellation only when Handy
  itself paused it.
- Existing media state is unchanged when the setting is off.

### Wave 8 — clipboard audit and tested merge

Owner: Windows/native clipboard agent. Independent, but must complete before Wave
3 is enabled by default (which will remain opt-in regardless).

Deliverables:

- Compare Handy's receipt-sequenced transaction with VoiceSnap's Windows
  implementation.
- Keep Handy's delayed-rendering, receipt, quiet-period, sequence-number, and
  ownership-loss design if it remains correct.
- Import only tested improvements:
  - STGMEDIUM-aware format materialization;
  - full-fidelity text/HTML/RTF/bitmap/file/custom format preservation where safe;
  - newer-owner-wins behavior;
  - delayed-rendering and clipboard-manager races;
  - cancellation and app-exit cleanup;
  - stress coverage over at least 1,000 paste/restore cycles.
- Do not replace clipboard paste with character-by-character Unicode typing.
- Keep default paste behavior unchanged until Word, Notepad, browsers, Office,
  Electron, Qt, and elevated-window cases pass manual smoke.

Acceptance:

- Normal paste remains as fast as the baseline after timing is measured.
- The user's clipboard is restored unless they explicitly chose to preserve the
  transcript.
- A newer user copy is never overwritten.
- Higher-integrity/UIPI failure produces an actionable fallback message.

## Parallel agent packets

Agents may work in separate branches/worktrees. Each packet must keep its scope
to the listed ownership areas and communicate any contract changes before
touching shared files.

There is one integration/contract steward. Only that steward may merge changes
to `settings.rs`, `actions.rs`, generated `bindings.ts`, and shared locale keys.
Feature agents may prepare isolated modules and contract patches, but must not
independently edit those chokepoints in parallel. Generated bindings are always
regenerated by the steward after the Rust command contract settles.

| Packet | Scope | Depends on | Main ownership |
|---|---|---|---|
| A0 | Visible diagnostics and timing contract | none | performance manager, diagnostics UI, event schema |
| A1 | Local cleanup engine/provider | A0 contract preferred | local provider, cleanup lifecycle, cleanup settings; shared settings/action wiring goes through steward |
| A2 | Streaming benchmark and overlay | A0 contract preferred | transcription stream events, overlay, benchmark fixtures |
| A3 | Committed-only live insertion | A0, A2, clipboard audit | insertion mode, session ledger, focus guard |
| B1 | Vocabulary and deterministic writing | none | isolated vocabulary/text modules; settings migration and shared UI wiring goes through steward |
| B2 | Rich history | A0, A1 | SQLite schema, history commands/UI |
| B3 | Hardware and microphone truth | A0 | hardware/runtime state, device recovery |
| B4 | Media pause/resume | none | media controller, async generation fencing, tests; setting wiring goes through steward |
| B5 | Windows clipboard audit | none | paste transaction, Windows fixtures/stress tests |
| C0 | Integration and release gates | all selected packets | build, smoke, packaging, docs |

The steward must serialize changes to the shared settings/action/IPC surface in
this order: timing contract, cleanup mode/provider contract, insertion mode,
vocabulary schema, history metadata, hardware/media settings. This avoids
parallel Specta/type-generation churn and makes each migration reviewable.

### Agent handoff protocol

Before starting a packet, an agent should record the packet ID, branch name,
files it expects to touch, and any shared contract it needs. A packet is ready
for integration only when its handoff includes:

- a short behavior summary;
- settings/database/IPC changes and migration notes;
- focused tests and their commands;
- known platform limitations;
- manual smoke steps;
- measured latency before/after when the packet touches the hot path;
- confirmation that no transcript/audio/clipboard contents were added to logs.

Agents should use conventional branch names such as
`feat/a0-visible-diagnostics`, `feat/a1-local-cleanup`, or
`fix/b5-windows-clipboard`. Keep commits focused. Do not rebase another agent's
branch or edit its worktree. The integration steward owns conflict resolution,
binding regeneration, combined tests, and merges into `main`.

The following files are likely collision points and need an explicit owner:

- `src-tauri/src/settings.rs`
- `src-tauri/src/actions.rs`
- `src-tauri/src/managers/transcription.rs`
- `src-tauri/src/managers/audio.rs`
- `src-tauri/src/managers/history.rs`
- `src-tauri/src/paste_tx/*`
- generated `src/bindings.ts`
- English and translated locale files

Do not have two agents independently redesign these files. Merge contract-first,
then implementation.

## Benchmark protocol

Before calling anything faster, run the same fixture and the same live phrases
through each candidate. Record:

- cold model load time;
- warm hotkey-to-capture-ready;
- first partial latency;
- committed-text cadence;
- stop-to-final-transcript;
- cleanup time;
- paste-visible time;
- total hotkey-to-idle;
- CPU utilization and peak RSS/VRAM where available;
- failures, cancellations, and audio drops;
- WER or a manually reviewed quality score on the fixed phrase set.

Use p50 and p95, not a single best run. Test short phrases (under 2 seconds),
normal dictation (5–15 seconds), and long speech (30–120 seconds). Test silence,
quiet speech, background audio, a device change, and a focus change.

## Required manual smoke matrix

At minimum, before enabling any new default:

- Windows 10 and Windows 11;
- no GPU, NVIDIA, AMD, and Intel Vulkan where available;
- Whisper Turbo final mode;
- one live Parakeet/Nemotron mode;
- local cleanup on and off;
- no-speech gate on, off, and shadow-only;
- Notepad, Word, browser, Chromium/Electron, terminal, and a code editor;
- normal paste, clipboard manager installed, newer copy during paste;
- focus change while recording;
- cancellation while streaming/finalizing/cleaning;
- microphone disappearance and recovery;
- media playing, media manually stopped, and media control unavailable.

## Release gates

No release candidate is complete until:

- `bun run lint` passes;
- `bun run build` passes;
- Rust formatting and clippy pass;
- focused unit tests pass;
- existing Handy tests pass;
- model catalog/license/attribution files are reviewed;
- visible Diagnostics shows real stage timings in a packaged build;
- normal final paste is unchanged or demonstrably better;
- live insertion remains opt-in;
- no-speech gate is off or shadow-only unless its false-negative acceptance
  tests and manual quiet-speech smoke are green;
- local cleanup has a working offline fallback;
- no raw speech content appears in logs or exported diagnostics.

## Explicitly deferred

- App-aware formatting profiles. They may be reconsidered later, but are not a
  dependency for cleanup, live streaming, vocabulary, or diagnostics.
- LAN/network transcription.
- Cloud STT and cloud streaming.
- File/audio-video transcription.
- Mouse side-button shortcuts.
- Onboarding wizard redesign.
- Generic accessibility-based document replacement.
- Automatic dual-ASR CPU/GPU decoding.
- Training a new speech or cleanup model.

## Source and licensing notes

Handy is MIT licensed. VoiceTypr is AGPL-3.0 licensed. This fork should preserve
Handy's MIT licensing unless the project intentionally makes a different legal
choice. Reimplement VoiceTypr's ideas; do not copy its source files into this
fork without a deliberate license review.

Useful references:

- [Handy](https://github.com/cjpais/Handy)
- [VoiceTypr](https://github.com/moinulmoin/voicetypr)
- [VoiceTypr writing settings](https://github.com/moinulmoin/voicetypr/blob/main/src-tauri/src/writing/settings.rs)
- [VoiceTypr vocabulary compiler](https://github.com/moinulmoin/voicetypr/blob/main/src-tauri/src/writing/vocabulary.rs)
- [VoiceTypr latency/streaming investigation](https://github.com/moinulmoin/voicetypr/blob/main/plans/028-transcription-latency-streaming.md)
- [VoiceTypr no-speech gate](https://github.com/moinulmoin/voicetypr/blob/main/plans/059-no-speech-gate.md)

## First implementation order

To get a useful fork running quickly:

1. Merge this plan and establish a clean baseline build.
2. Implement A0 visible diagnostics first.
3. In parallel, implement B1 vocabulary, B4 media pause, and B5 clipboard audit.
4. Implement A1 local cleanup and A2 streaming benchmark/overlay.
5. Implement B2 history and B3 hardware truth using the established timing schema.
6. Implement A3 live insertion last, behind its setting and warning.
7. Run C0 integration, packaged smoke, and the benchmark matrix.

The first milestone is not “every feature is done.” It is: the fork can run a
warm live model, show where every millisecond goes, and safely paste a final
result. Every later optimization should be justified by the visible data.
