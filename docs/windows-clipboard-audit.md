# Windows clipboard audit

This audit covers the receipt-sequenced Windows paste transaction in `src-tauri/src/paste_tx/windows.rs` and the remaining Wave 8 format-preservation requirements.

## Reference comparison

The reference inspected was VoiceSnap commit `4ba961943957e3e67dfdf2722ffee23600eb4b2c`, `VoiceSnapGo/internal/input/paste_windows.go`.

VoiceSnap's normal `Paste` path:

- snapshots only `CF_UNICODETEXT`;
- replaces the clipboard with transcript text;
- sends Ctrl+V with `SendInput`;
- sleeps 300 ms on a goroutine and then restores the old text;
- has no clipboard sequence-number or ownership fence around that delayed restore.

VoiceSnap also has a separate per-character Unicode `TypeText` implementation. That is not an improvement for Handy's normal paste path and is deliberately not imported.

Handy's receipt-sequenced transaction is retained because it closes correctness gaps in the reference approach:

- `SetClipboardData(CF_UNICODETEXT, NULL)` delays rendering until a consumer requests the transcript;
- `WM_RENDERFORMAT` is treated as the read receipt, and only post-injection receipts count;
- a short quiet period covers consumers that probe/read more than once;
- restore/preserve settlement checks clipboard sequence and ownership while the clipboard is open, so a newer user/app copy always wins;
- normal settlement restores the previous clipboard, while `ClipboardHandling::CopyToClipboard` explicitly leaves the transcript;
- a failed synthetic input path restores synchronously when still owned and reports the likely UIPI/integrity-level problem.

A fixed restore delay would regress these guarantees, so the VoiceSnap timing model is not adopted.

## STGMEDIUM-aware preservation policy

Clipboard handles cannot safely be treated as interchangeable `HGLOBAL` values. The audit keeps the preservation set explicit and medium-aware:

| Representative format | Requested/copied medium | Policy |
| --- | --- | --- |
| `CF_UNICODETEXT` / ordinary text | `TYMED_HGLOBAL` | Copy complete global-memory payload |
| registered HTML (`HTML Format`) | `TYMED_HGLOBAL` when the data object can materialize it | Preserve |
| registered RTF (`Rich Text Format`) | `TYMED_HGLOBAL` when materializable | Preserve |
| `CF_HDROP` file list | `TYMED_HGLOBAL` | Preserve complete DROPFILES payload |
| registered custom flat formats | `TYMED_HGLOBAL` when materializable | Preserve |
| `CF_BITMAP` | `TYMED_GDI` | Duplicate with `CopyImage`; transfer duplicate on restore |
| DIB/DIBV5 and other safe flat formats | `TYMED_HGLOBAL` when materializable | Preserve |
| metafile, palette, owner-display and display-only formats | non-flat/special ownership | Skip rather than reinterpret the handle |
| stream/storage-only OLE formats | no safe requested `TYMED_HGLOBAL` | Skip |

`IDataObject::GetData` performs delayed-format materialization. Every returned `STGMEDIUM` is released with `ReleaseStgMedium`; bitmap copies are independently owned until either transferred back to the clipboard or deleted.

The code intentionally does not coerce a stream/storage/metafile medium into bytes. Losing one unsupported special format is safer than corrupting it or violating its ownership contract.

## Automated coverage

Windows-specific tests cover:

- safe medium classification for text, HTML, RTF, file-drop, bitmap and registered custom formats;
- actual `HGLOBAL` allocation/lock/copy materialization through the production byte-copy helper;
- 1,000 normal paste/restore cycles and 1,000 forced newer-owner races;
- normal restore versus explicit transcript preservation, including newer-owner-wins behavior;
- delayed-render/clipboard-manager receipt races through the shared transaction state machine;
- UIPI/input-failure restoration and actionable error text;
- a normal receipt-sequenced transaction timing report plus a separate format-preservation bookkeeping regression guard.

### Automated timing method and threshold

`normal_receipt_sequenced_transaction_timing_stays_below_regression_budget` runs 20,000 deterministic normal transactions through the Windows test harness. Each cycle executes the normal semantic path: snapshot/publish, a post-injection delayed-render receipt, the quiet-period decision, sequence-fenced settlement, and restoration of the prior clipboard. The receipt timestamp is advanced by `QUIET_PERIOD` instead of sleeping, so the measurement captures Handy's transaction CPU cost rather than scheduler or target-application event-loop latency.

The test reports only cycle count, total elapsed time, mean elapsed harness time per transaction, and the threshold; it never prints clipboard fixture or transcript contents. The explicit material-regression threshold is **1 millisecond mean transaction time** across 20,000 cycles. This is intentionally much larger than the expected in-memory bookkeeping cost so shared-runner jitter does not make the guard flaky, while still catching a regression that adds millisecond-scale work to every normal paste transaction.

The existing format-preservation guard remains separate. It compares the prior text-only transaction bookkeeping with representative rich-format snapshot bookkeeping over 20,000 cycles and permits at most 500 microseconds of added CPU work per transaction. Both timing checks intentionally exclude `SendInput`, the unchanged 100 ms chord hold, and real waiting during the receipt quiet period: those are unchanged/default-path behavior or external application scheduling, not preservation bookkeeping. The timing harness verifies the quiet-period branch using monotonic timestamps without changing the production constants.

The normal path remains a clipboard paste chord (`Ctrl+V`, `Ctrl+Shift+V`, or `Shift+Insert`); no character-by-character Unicode typing was substituted and no default paste setting is changed by the benchmark.

## Manual gate

These automated checks do not replace the PLAN manual smoke gate. Before the receipt-sequenced path becomes a new default, Windows 10/11 testing should still cover Notepad, Word/Office, browsers, Chromium/Electron, Qt, terminals/code editors, clipboard managers, newer-copy races and elevated-window/UIPI cases.
