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
- UIPI/input-failure restoration and actionable error text;
- a timing regression guard for the additional format-preservation bookkeeping.

The timing guard compares the prior text-only transaction bookkeeping with representative rich-format snapshot bookkeeping over 20,000 cycles. The allowed added CPU budget is 500 microseconds per transaction. It intentionally excludes `SendInput`, the unchanged 100 ms chord hold, and receipt quiet-period timing, so it detects an accidental expensive preservation loop without pretending to benchmark target-application event-loop latency.

The normal path remains a clipboard paste chord (`Ctrl+V`, `Ctrl+Shift+V`, or `Shift+Insert`); no character-by-character Unicode typing was substituted.

## Manual gate

These automated checks do not replace the PLAN manual smoke gate. Before the receipt-sequenced path becomes a new default, Windows 10/11 testing should still cover Notepad, Word/Office, browsers, Chromium/Electron, Qt, terminals/code editors, clipboard managers, newer-copy races and elevated-window/UIPI cases.
