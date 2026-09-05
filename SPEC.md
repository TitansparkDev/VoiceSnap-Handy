# VoiceSnap-Handy

VoiceSnap-Handy is a local-first desktop speech-to-text app based on Handy.
It records from a selected microphone, transcribes speech locally, optionally
cleans the result, and pastes the final text into the active application.

## Current product

- Platforms: Windows 10/11, macOS, and supported Linux distributions.
- Default flow: press the global shortcut, speak, stop, and paste one final
  transcription.
- Streaming-capable models can show committed and tentative text in the live
  overlay without inserting tentative text.
- Optional experimental live insertion inserts only stable committed deltas.
- Settings include shortcuts, audio devices, models, language, feedback,
  cleanup, vocabulary, diagnostics, history, and media pause behavior.
- History is stored locally with searchable raw and processed text.
- Diagnostics store bounded, privacy-safe timing and lifecycle metadata only.

## Safety and privacy

Transcription, cleanup, audio capture, clipboard handling, history, and
diagnostics are local by default. Tentative text is never pasted. Clipboard
contents, audio, transcripts, window titles, and document text are not written
to diagnostics by default. Live insertion is opt-in and visibly experimental.
