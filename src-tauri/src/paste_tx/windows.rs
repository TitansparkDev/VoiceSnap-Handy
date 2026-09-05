//! Windows reliable paste.
//!
//! Publishes the transcript as a *delayed-render* clipboard format
//! (`SetClipboardData(CF_UNICODETEXT, NULL)`) owned by a hidden message-only
//! window. Windows sends the owner `WM_RENDERFORMAT` when a consumer actually
//! requests the data — that message is the read receipt. The previous
//! clipboard contents (snapshotted with full format fidelity) are restored
//! once receipts go quiet (see `paste_tx::evaluate`), guarded by the clipboard
//! sequence number so we never clobber a newer user copy.
//!
//! Threading: clipboard ownership and delayed rendering are per-thread and
//! need a message pump, so the whole transaction lives on a dedicated worker
//! thread. The calling thread only sends the paste chord once the worker
//! signals the transcript is published, then returns; the wait, guarded
//! restore and auto-submit all finish on the worker.

use std::sync::{mpsc::Sender, Arc, Mutex, Once};
use std::thread;
use std::time::Instant;

use log::{error, info, warn};
use tauri::Manager;
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{
    SetLastError, ERROR_SUCCESS, HANDLE, HGLOBAL, HINSTANCE, HWND, LPARAM, LRESULT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{DeleteObject, HGDIOBJ};

use super::{evaluate, may_restore, send_chord, ReliablePasteError, TxState, WaitDecision};
use crate::clipboard::send_return_key;
use crate::input::EnigoState;
use crate::settings::{AutoSubmitKey, ClipboardHandling, PasteMethod};
use windows::Win32::Foundation::GlobalFree;
use windows::Win32::System::Com::{
    IDataObject, DVASPECT_CONTENT, FORMATETC, TYMED_GDI, TYMED_HGLOBAL,
};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardOwner,
    GetClipboardSequenceNumber, OpenClipboard, RegisterClipboardFormatW, SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{
    GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
};
use windows::Win32::System::Ole::{
    OleGetClipboard, OleInitialize, OleUninitialize, ReleaseStgMedium, CF_BITMAP, CF_DSPBITMAP,
    CF_DSPENHMETAFILE, CF_DSPMETAFILEPICT, CF_DSPTEXT, CF_ENHMETAFILE, CF_OWNERDISPLAY, CF_PALETTE,
    CF_UNICODETEXT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CopyImage, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW,
    GetWindowLongPtrW, KillTimer, PostQuitMessage, RegisterClassW, SetTimer, SetWindowLongPtrW,
    GDI_IMAGE_TYPE, GWLP_USERDATA, HWND_MESSAGE, IMAGE_FLAGS, MSG, WINDOW_EX_STYLE, WINDOW_STYLE,
    WM_DESTROYCLIPBOARD, WM_RENDERALLFORMATS, WM_RENDERFORMAT, WM_TIMER, WNDCLASSW,
};

const CLASS_NAME: PCWSTR = w!("HandyPasteTxWindow");
const TIMER_ID: usize = 1;
const TIMER_INTERVAL_MS: u32 = 25;
/// Skip clipboard formats larger than this when snapshotting.
const MAX_FORMAT_BYTES: usize = 64 * 1024 * 1024;

const IMAGE_BITMAP_TYPE: GDI_IMAGE_TYPE = GDI_IMAGE_TYPE(0);
const LR_CREATEDIBSECTION_FLAG: IMAGE_FLAGS = IMAGE_FLAGS(0x2000);

struct OleGuard;

impl OleGuard {
    unsafe fn initialize() -> Result<Self, String> {
        OleInitialize(None).map_err(|e| format!("OleInitialize failed: {e}"))?;
        Ok(Self)
    }
}

impl Drop for OleGuard {
    fn drop(&mut self) {
        unsafe { OleUninitialize() };
    }
}

struct SavedFormat {
    format: u32,
    data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotMediumKind {
    HGlobal,
    GdiBitmap,
    Skip,
}

/// Clipboard formats are typed by storage medium, not by HANDLE shape. Keep the
/// safe set deliberately narrow: flat HGLOBAL data can be byte-copied, and a
/// CF_BITMAP returned as TYMED_GDI can be duplicated with CopyImage. Metafiles,
/// palettes, owner-display and display-only formats have different ownership /
/// lifetime rules and are intentionally left to the current clipboard owner.
fn snapshot_medium_kind(format: u32) -> SnapshotMediumKind {
    if format == CF_BITMAP.0 as u32 {
        SnapshotMediumKind::GdiBitmap
    } else if format == CF_ENHMETAFILE.0 as u32
        || format == CF_DSPENHMETAFILE.0 as u32
        || format == CF_DSPBITMAP.0 as u32
        || format == CF_DSPMETAFILEPICT.0 as u32
        || format == CF_DSPTEXT.0 as u32
        || format == CF_OWNERDISPLAY.0 as u32
        || format == CF_PALETTE.0 as u32
    {
        SnapshotMediumKind::Skip
    } else {
        SnapshotMediumKind::HGlobal
    }
}

pub(super) struct WinTxShared {
    state: Mutex<TxState>,
    text: String,
    snapshot: Mutex<Vec<SavedFormat>>,
    /// Copied HBITMAP (as raw usize), restored via SetClipboardData.
    saved_bitmap: Mutex<Option<usize>>,
    sequence: Mutex<u32>,
    app_handle: tauri::AppHandle,
    auto_submit: bool,
    auto_submit_key: AutoSubmitKey,
    /// ClipboardHandling::CopyToClipboard — settle by leaving the transcript
    /// on the clipboard as plain text instead of restoring the snapshot.
    preserve_transcript: bool,
}

/// The transaction currently holding the clipboard, if any. A new
/// transaction settles it before snapshotting (see `flush_pending`).
static PENDING: Mutex<Option<Arc<WinTxShared>>> = Mutex::new(None);

fn discard_saved_bitmap(shared: &WinTxShared) {
    if let Ok(mut bitmap) = shared.saved_bitmap.lock() {
        if let Some(raw) = bitmap.take() {
            unsafe {
                let _ = DeleteObject(HGDIOBJ(raw as *mut _));
            }
        }
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

unsafe fn shared_ptr(hwnd: HWND) -> *const WinTxShared {
    GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const WinTxShared
}

/// Sends the auto-submit Enter. Uses `try_lock` because the paste caller may
/// currently hold the enigo lock while waiting for this worker.
fn send_auto_submit(shared: &WinTxShared) {
    {
        let mut st = match shared.state.lock() {
            Ok(st) => st,
            Err(_) => return,
        };
        if st.auto_submit_sent {
            return;
        }
        st.auto_submit_sent = true;
    }
    if let Some(enigo_state) = shared.app_handle.try_state::<EnigoState>() {
        match enigo_state.0.try_lock() {
            Ok(mut enigo) => {
                let _ = send_return_key(&mut enigo, shared.auto_submit_key);
            }
            Err(_) => warn!("[reliable-paste] skipping auto-submit: input state busy"),
        }
    }
}

/// Renders the promised transcript into the clipboard, which must already be
/// open: the system opens it on our behalf for WM_RENDERFORMAT; every other
/// caller has to wrap this in OpenClipboard/CloseClipboard itself.
unsafe fn render_text(shared: &WinTxShared) {
    let wide_text: Vec<u16> = shared
        .text
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let Ok(hg) = GlobalAlloc(GMEM_MOVEABLE, wide_text.len() * 2) else {
        return;
    };
    let ptr = GlobalLock(hg) as *mut u16;
    if ptr.is_null() {
        let _ = GlobalFree(Some(hg));
        return;
    }
    std::ptr::copy_nonoverlapping(wide_text.as_ptr(), ptr, wide_text.len());
    let _ = GlobalUnlock(hg);
    if SetClipboardData(CF_UNICODETEXT.0 as u32, Some(HANDLE(hg.0))).is_err() {
        let _ = GlobalFree(Some(hg));
    }
}

unsafe extern "system" fn paste_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let shared = shared_ptr(hwnd);
    match msg {
        WM_RENDERFORMAT => {
            if !shared.is_null() {
                let shared = &*shared;
                if let Ok(mut st) = shared.state.lock() {
                    st.record_receipt(Instant::now());
                }
                if wparam.0 as u32 == CF_UNICODETEXT.0 as u32 {
                    render_text(shared);
                }
            }
            LRESULT(0)
        }
        WM_RENDERALLFORMATS => {
            // Sent when the window is destroyed while an unrendered promise is
            // still on the clipboard — not a consumer read, so no receipt.
            // Unlike WM_RENDERFORMAT the system does not open the clipboard on
            // our behalf here: open it and confirm we still own it first.
            if !shared.is_null() {
                let shared = &*shared;
                if OpenClipboard(Some(hwnd)).is_ok() {
                    if GetClipboardOwner()
                        .map(|owner| owner == hwnd)
                        .unwrap_or(false)
                    {
                        render_text(shared);
                    }
                    let _ = CloseClipboard();
                }
            }
            LRESULT(0)
        }
        WM_DESTROYCLIPBOARD => {
            if !shared.is_null() {
                if let Ok(mut st) = (&*shared).state.lock() {
                    st.ownership_lost = true;
                }
            }
            LRESULT(0)
        }
        WM_TIMER => {
            if !shared.is_null() {
                on_timer(hwnd, &*shared);
            }
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

fn ensure_window_class(hinstance: HINSTANCE) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let wc = WNDCLASSW {
            lpfnWndProc: Some(paste_wnd_proc),
            hInstance: hinstance,
            lpszClassName: CLASS_NAME,
            ..Default::default()
        };
        unsafe {
            RegisterClassW(&wc);
        }
    });
}

/// If a previous transaction is still holding the clipboard, settle it now so
/// the snapshot below captures the user's original clipboard content. The
/// previous worker observes `cancelled` on its next timer tick and tears down
/// without restoring.
fn flush_pending() {
    let previous = match PENDING.lock() {
        Ok(mut slot) => slot.take(),
        Err(_) => None,
    };
    let Some(previous) = previous else {
        return;
    };
    let (receipt, claimed_settlement) = {
        let mut st = match previous.state.lock() {
            Ok(st) => st,
            Err(_) => return,
        };
        st.cancelled = true;
        let receipt = st.any_receipt_after_injection();
        (receipt, st.claim_settlement())
    };
    if !claimed_settlement {
        return;
    }
    if previous.auto_submit && receipt {
        send_auto_submit(&previous);
    }
    let sequence = *previous.sequence.lock().unwrap();
    if !unsafe { settle_clipboard(&previous, sequence) } {
        discard_saved_bitmap(&previous);
    }
}

pub(super) fn shutdown_pending() {
    // App exit is just an explicit cancellation of the active transaction:
    // restore only if our sequence is still current, otherwise the newer owner
    // wins and its clipboard contents remain untouched.
    flush_pending();
}

fn clear_pending_if_same(shared: &WinTxShared) {
    if let Ok(mut slot) = PENDING.lock() {
        let is_us = slot
            .as_ref()
            .map(|pending| Arc::as_ptr(pending) as *const WinTxShared == shared as *const _)
            .unwrap_or(false);
        if is_us {
            *slot = None;
        }
    }
}

/// A failed SendInput/enigo chord on Windows is commonly UIPI: a normal-integrity
/// Handy cannot inject into an elevated target. Settle synchronously so the
/// caller gets an actionable error only after the original clipboard has been
/// restored when still owned by this transaction. If a newer owner already won,
/// the sequence fence leaves that clipboard content untouched.
fn abort_after_input_failure(shared: &WinTxShared) {
    let claimed_settlement = {
        let mut st = match shared.state.lock() {
            Ok(st) => st,
            Err(_) => return,
        };
        st.injection_failed = true;
        st.cancelled = true;
        st.claim_settlement()
    };

    clear_pending_if_same(shared);
    if !claimed_settlement {
        return;
    }

    let sequence = *shared.sequence.lock().unwrap();
    if !unsafe { restore_snapshot_if_current(shared, sequence) } {
        info!("[reliable-paste] input failed after clipboard changed externally; leaving newer content untouched");
    }
    discard_saved_bitmap(shared);
}

/// Settle-time clipboard handling. The sequence/ownership fence is checked
/// *after* opening the clipboard; once it is open no newer owner can slip in
/// between the check and EmptyClipboard/restore.
unsafe fn settle_clipboard(shared: &WinTxShared, expected_sequence: u32) -> bool {
    if !shared.preserve_transcript {
        let restored = restore_snapshot_if_current(shared, expected_sequence);
        discard_saved_bitmap(shared);
        return restored;
    }
    if OpenClipboard(None).is_err() {
        warn!("[reliable-paste] could not open clipboard to leave transcript");
        return false;
    }
    let current_sequence = GetClipboardSequenceNumber();
    let still_ours = shared
        .state
        .lock()
        .map(|st| may_restore(&st, expected_sequence, current_sequence))
        .unwrap_or(false);
    if !still_ours {
        let _ = CloseClipboard();
        return false;
    }
    let _ = EmptyClipboard();
    render_text(shared);
    let _ = CloseClipboard();
    discard_saved_bitmap(shared);
    info!("[reliable-paste] left transcript on clipboard as plain text");
    true
}

/// Restores the snapshotted clipboard contents while the clipboard is already
/// open. This is also used to roll back a partially-published transaction before
/// releasing the clipboard lock, so another owner can never slip in between a
/// failed publish and our restoration.
unsafe fn restore_snapshot_open(shared: &WinTxShared) {
    let _ = EmptyClipboard();
    if let Ok(formats) = shared.snapshot.lock() {
        for saved in formats.iter() {
            if saved.data.is_empty() {
                continue;
            }
            let Ok(hg) = GlobalAlloc(GMEM_MOVEABLE, saved.data.len()) else {
                continue;
            };
            let ptr = GlobalLock(hg) as *mut u8;
            if ptr.is_null() {
                let _ = GlobalFree(Some(hg));
                continue;
            }
            std::ptr::copy_nonoverlapping(saved.data.as_ptr(), ptr, saved.data.len());
            let _ = GlobalUnlock(hg);
            // SetClipboardData takes ownership of the handle on success.
            if SetClipboardData(saved.format, Some(HANDLE(hg.0))).is_err() {
                let _ = GlobalFree(Some(hg));
            }
        }
    }
    if let Ok(mut bitmap) = shared.saved_bitmap.lock() {
        if let Some(raw) = bitmap.take() {
            let _ = SetClipboardData(CF_BITMAP.0 as u32, Some(HANDLE(raw as *mut _)));
        }
    }
}

/// Restores the snapshotted clipboard only while this transaction is still the
/// current clipboard owner. The sequence check happens while the clipboard is
/// open, closing the check-then-open race against a newer user copy.
unsafe fn restore_snapshot_if_current(shared: &WinTxShared, expected_sequence: u32) -> bool {
    if OpenClipboard(None).is_err() {
        warn!("[reliable-paste] could not open clipboard to restore");
        return false;
    }
    let current_sequence = GetClipboardSequenceNumber();
    let still_ours = shared
        .state
        .lock()
        .map(|st| may_restore(&st, expected_sequence, current_sequence))
        .unwrap_or(false);
    if !still_ours {
        let _ = CloseClipboard();
        return false;
    }
    restore_snapshot_open(shared);
    let _ = CloseClipboard();
    info!("[reliable-paste] restored previous clipboard");
    true
}

unsafe fn copy_hglobal_bytes(hg: HGLOBAL) -> Option<Vec<u8>> {
    let size = GlobalSize(hg);
    if size == 0 || size > MAX_FORMAT_BYTES {
        return None;
    }
    let ptr = GlobalLock(hg) as *const u8;
    if ptr.is_null() {
        return None;
    }
    let copied = std::slice::from_raw_parts(ptr, size).to_vec();
    let _ = GlobalUnlock(hg);
    Some(copied)
}

unsafe fn copy_hglobal_medium(data_object: &IDataObject, format: u32) -> Option<Vec<u8>> {
    let request = FORMATETC {
        cfFormat: format as u16,
        ptd: std::ptr::null_mut(),
        dwAspect: DVASPECT_CONTENT.0,
        lindex: -1,
        tymed: TYMED_HGLOBAL.0 as u32,
    };
    let mut medium = data_object.GetData(&request).ok()?;
    let data = if medium.tymed == TYMED_HGLOBAL.0 as u32 {
        copy_hglobal_bytes(medium.u.hGlobal)
    } else {
        None
    };
    ReleaseStgMedium(&mut medium);
    data
}

unsafe fn copy_bitmap_medium(data_object: &IDataObject) -> Option<usize> {
    let request = FORMATETC {
        cfFormat: CF_BITMAP.0 as u16,
        ptd: std::ptr::null_mut(),
        dwAspect: DVASPECT_CONTENT.0,
        lindex: -1,
        tymed: TYMED_GDI.0 as u32,
    };
    let mut medium = data_object.GetData(&request).ok()?;
    let copy = if medium.tymed == TYMED_GDI.0 as u32 {
        CopyImage(
            HANDLE(medium.u.hBitmap.0),
            IMAGE_BITMAP_TYPE,
            0,
            0,
            LR_CREATEDIBSECTION_FLAG,
        )
        .ok()
        .map(|handle| handle.0 as usize)
    } else {
        None
    };
    ReleaseStgMedium(&mut medium);
    copy
}

unsafe fn snapshot_clipboard(hwnd: HWND, shared: &WinTxShared) -> Result<u32, String> {
    let start_owner = GetClipboardOwner().ok();
    OpenClipboard(Some(hwnd)).map_err(|e| format!("OpenClipboard failed: {e}"))?;
    let mut available_formats = Vec::new();
    let mut format = 0u32;
    loop {
        format = EnumClipboardFormats(format);
        if format == 0 {
            break;
        }
        available_formats.push(format);
    }
    CloseClipboard().map_err(|e| format!("CloseClipboard failed: {e}"))?;

    // Ask the OLE data object for explicit media rather than assuming every
    // clipboard HANDLE is HGLOBAL. Requesting TYMED_HGLOBAL safely materializes
    // normal text, HTML/RTF, CF_HDROP and registered custom formats that can be
    // represented as flat global memory; formats backed only by streams,
    // storage, metafiles, palettes, owner-display, etc. are left untouched.
    let data_object = OleGetClipboard().map_err(|e| format!("OleGetClipboard failed: {e}"))?;
    let mut formats = Vec::new();
    for format in available_formats {
        match snapshot_medium_kind(format) {
            SnapshotMediumKind::GdiBitmap => {
                if let Some(copy) = copy_bitmap_medium(&data_object) {
                    if let Ok(mut slot) = shared.saved_bitmap.lock() {
                        *slot = Some(copy);
                    }
                }
            }
            SnapshotMediumKind::HGlobal => {
                if let Some(data) = copy_hglobal_medium(&data_object, format) {
                    formats.push(SavedFormat { format, data });
                }
            }
            SnapshotMediumKind::Skip => {}
        }
    }

    let end_owner = GetClipboardOwner().ok();
    if end_owner != start_owner {
        return Err(
            "clipboard owner changed while snapshotting; preserving newer owner".to_string(),
        );
    }
    // Delayed rendering itself can increment the clipboard sequence number, so
    // record the post-materialization sequence rather than treating that change
    // as external ownership. The publish-side sequence fence below rejects any
    // change that occurs after the snapshot is fully materialized.
    let end_sequence = GetClipboardSequenceNumber();
    if let Ok(mut slot) = shared.snapshot.lock() {
        *slot = formats;
    }
    Ok(end_sequence)
}

/// Publishes the transcript as a delayed-render promise plus clipboard
/// history / cloud / monitoring opt-out markers (the same formats Chrome uses
/// for Incognito copies). Returns the new clipboard sequence number.
unsafe fn publish(hwnd: HWND, shared: &WinTxShared, expected_sequence: u32) -> Result<u32, String> {
    OpenClipboard(Some(hwnd)).map_err(|e| format!("OpenClipboard failed: {e}"))?;
    if GetClipboardSequenceNumber() != expected_sequence {
        let _ = CloseClipboard();
        return Err(
            "clipboard changed before transcript publish; preserving newer owner".to_string(),
        );
    }

    let published = publish_formats();
    if let Err(error) = published {
        // publish_formats may already have emptied the clipboard. Roll back
        // while we still hold the clipboard lock, before any newer owner can
        // acquire it.
        restore_snapshot_open(shared);
        let _ = CloseClipboard();
        return Err(error);
    }
    CloseClipboard().map_err(|e| format!("CloseClipboard failed: {e}"))?;
    Ok(GetClipboardSequenceNumber())
}

/// Everything `publish` does while the clipboard is open, split out so
/// `publish` closes the clipboard on every path — bailing out while holding it
/// open (and possibly already emptied) would strand the clipboard and leave
/// the legacy fallback snapshotting nothing.
unsafe fn publish_formats() -> Result<(), String> {
    EmptyClipboard().map_err(|e| format!("EmptyClipboard failed: {e}"))?;

    for (name, value) in [
        ("ExcludeClipboardContentFromMonitorProcessing", 1u32),
        ("CanIncludeInClipboardHistory", 0u32),
        ("CanUploadToCloudClipboard", 0u32),
    ] {
        let name_wide = wide(name);
        let format = RegisterClipboardFormatW(PCWSTR(name_wide.as_ptr()));
        if format == 0 {
            continue;
        }
        if let Ok(hg) = GlobalAlloc(GMEM_MOVEABLE, std::mem::size_of::<u32>()) {
            let ptr = GlobalLock(hg) as *mut u32;
            if !ptr.is_null() {
                *ptr = value;
                let _ = GlobalUnlock(hg);
                if SetClipboardData(format, Some(HANDLE(hg.0))).is_err() {
                    let _ = GlobalFree(Some(hg));
                }
            } else {
                let _ = GlobalFree(Some(hg));
            }
        }
    }

    // NULL handle = delayed rendering: we are only asked for the data (via
    // WM_RENDERFORMAT) when a consumer actually reads it. SetClipboardData
    // returns the handle it was given, so for delayed rendering success is
    // also NULL and the windows crate reports it as an Err carrying
    // GetLastError(). Only a nonzero thread error is a real failure, and the
    // thread error must be cleared first so a stale value from an earlier
    // call can't masquerade as one.
    SetLastError(ERROR_SUCCESS);
    if let Err(e) = SetClipboardData(CF_UNICODETEXT.0 as u32, None) {
        if e.code().is_err() {
            return Err(format!("SetClipboardData failed: {e}"));
        }
    }
    Ok(())
}

fn on_timer(_hwnd: HWND, shared: &WinTxShared) {
    let now = Instant::now();
    let finish = {
        let mut st = match shared.state.lock() {
            Ok(st) => st,
            Err(_) => return,
        };
        if st.cancelled {
            true
        } else {
            match evaluate(&st, now) {
                WaitDecision::KeepWaiting => false,
                WaitDecision::Finish => {
                    st.cancelled = true;
                    true
                }
            }
        }
    };
    if !finish {
        return;
    }

    let (receipt, ownership_lost, injection_failed, claimed_settlement) = {
        let mut st = match shared.state.lock() {
            Ok(st) => st,
            Err(_) => return,
        };
        let receipt = st.any_receipt_after_injection();
        let ownership_lost = st.ownership_lost;
        let injection_failed = st.injection_failed;
        let claimed_settlement = st.claim_settlement();
        (
            receipt,
            ownership_lost,
            injection_failed,
            claimed_settlement,
        )
    };
    if !claimed_settlement {
        clear_pending_if_same(shared);
        unsafe { PostQuitMessage(0) };
        return;
    }
    if ownership_lost {
        info!("[reliable-paste] settling: clipboard ownership lost");
    } else if receipt {
        info!("[reliable-paste] settling: reads went quiet");
    } else if injection_failed {
        info!("[reliable-paste] settling: chord injection failed, restoring quickly");
    } else {
        info!("[reliable-paste] settling: no read within timeout, restoring anyway");
    }

    // Auto-submit only once the target demonstrably read the transcript;
    // pressing Enter after an unconfirmed paste could submit stale content.
    if shared.auto_submit && receipt {
        send_auto_submit(shared);
    }

    let sequence = *shared.sequence.lock().unwrap();
    if !unsafe { settle_clipboard(shared, sequence) } {
        discard_saved_bitmap(shared);
        info!("[reliable-paste] clipboard changed externally; leaving it untouched");
    }

    clear_pending_if_same(shared);

    unsafe {
        PostQuitMessage(0);
    }
}

unsafe fn destroy_window_and_shared(hwnd: HWND) {
    let ptr = shared_ptr(hwnd);
    let _ = DestroyWindow(hwnd);
    if !ptr.is_null() {
        drop(Arc::from_raw(ptr));
    }
}

fn pump_thread(shared: Arc<WinTxShared>, ready: Sender<Result<(), String>>) {
    unsafe {
        let _ole = match OleGuard::initialize() {
            Ok(guard) => guard,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };

        // Settle any previous transaction first so the snapshot captures the
        // user's original clipboard, not the previous transcript.
        flush_pending();

        let hinstance = match GetModuleHandleW(PCWSTR::null()) {
            Ok(hmodule) => HINSTANCE(hmodule.0),
            Err(e) => {
                let _ = ready.send(Err(format!("GetModuleHandle failed: {e}")));
                return;
            }
        };
        ensure_window_class(hinstance);

        let hwnd = match CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            CLASS_NAME,
            w!("HandyPasteTx"),
            WINDOW_STYLE::default(),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            Some(hinstance),
            None,
        ) {
            Ok(hwnd) => hwnd,
            Err(e) => {
                let _ = ready.send(Err(format!("CreateWindowEx failed: {e}")));
                return;
            }
        };
        SetWindowLongPtrW(
            hwnd,
            GWLP_USERDATA,
            Arc::into_raw(shared.clone()) as *const _ as isize,
        );

        let published = match snapshot_clipboard(hwnd, &shared) {
            Ok(snapshot_sequence) => publish(hwnd, &shared, snapshot_sequence),
            Err(e) => Err(e),
        };
        let sequence = match published {
            Ok(sequence) => sequence,
            Err(e) => {
                // A bitmap clone is process-owned until restoration transfers
                // it to the clipboard. Drop it on every aborted transaction.
                discard_saved_bitmap(&shared);
                destroy_window_and_shared(hwnd);
                let _ = ready.send(Err(e));
                return;
            }
        };
        *shared.sequence.lock().unwrap() = sequence;
        shared.state.lock().unwrap().published_at = Instant::now();
        if let Ok(mut slot) = PENDING.lock() {
            *slot = Some(shared.clone());
        }
        let _ = SetTimer(Some(hwnd), TIMER_ID, TIMER_INTERVAL_MS, None);
        let _ = ready.send(Ok(()));

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = DispatchMessageW(&msg);
        }

        let _ = KillTimer(Some(hwnd), TIMER_ID);
        destroy_window_and_shared(hwnd);
    }
}

pub(super) fn run(
    text: &str,
    app_handle: &tauri::AppHandle,
    paste_method: &PasteMethod,
    enigo: &mut enigo::Enigo,
    auto_submit: bool,
    auto_submit_key: AutoSubmitKey,
    clipboard_handling: ClipboardHandling,
) -> Result<(), ReliablePasteError> {
    let shared = Arc::new(WinTxShared {
        state: Mutex::new(TxState::new()),
        text: text.to_string(),
        snapshot: Mutex::new(Vec::new()),
        saved_bitmap: Mutex::new(None),
        sequence: Mutex::new(0),
        app_handle: app_handle.clone(),
        auto_submit,
        auto_submit_key,
        preserve_transcript: clipboard_handling == ClipboardHandling::CopyToClipboard,
    });

    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let shared_for_pump = shared.clone();
    thread::spawn(move || pump_thread(shared_for_pump, ready_tx));

    // Wait until the transcript is actually published (or the worker reports
    // why it could not) before injecting the chord.
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(ReliablePasteError::Unavailable(e)),
        Err(_) => {
            return Err(ReliablePasteError::Unavailable(
                "reliable paste worker died before publishing".to_string(),
            ))
        }
    }
    info!("[reliable-paste] published transcript (delayed render)");

    // Mark injection *before* sending: enigo holds the chord for ~100ms and a
    // fast target may legitimately read while the chord is still held.
    shared.state.lock().unwrap().injected_at = Some(Instant::now());
    match send_chord(enigo, paste_method) {
        Ok(()) => {
            info!("[reliable-paste] paste chord sent ({paste_method:?})");
        }
        Err(e) => {
            error!("[reliable-paste] failed to send paste chord: {e}");
            abort_after_input_failure(&shared);
            return Err(ReliablePasteError::WindowsInputBlocked(e));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRESS_CYCLES: usize = 1_000;

    #[derive(Debug)]
    struct HarnessClipboard {
        contents: String,
        sequence: u32,
    }

    impl HarnessClipboard {
        fn new(contents: impl Into<String>) -> Self {
            Self {
                contents: contents.into(),
                sequence: 1,
            }
        }

        fn replace(&mut self, contents: impl Into<String>) {
            self.contents = contents.into();
            self.sequence = self.sequence.wrapping_add(1);
        }
    }

    struct HarnessTx {
        snapshot: String,
        transcript: String,
        published_sequence: u32,
        state: TxState,
    }

    impl HarnessTx {
        fn publish(clipboard: &mut HarnessClipboard, transcript: String) -> Self {
            let snapshot = clipboard.contents.clone();
            clipboard.replace(transcript.clone());
            let mut state = TxState::new();
            state.injected_at = Some(Instant::now());
            Self {
                snapshot,
                transcript,
                published_sequence: clipboard.sequence,
                state,
            }
        }

        fn settle(&mut self, clipboard: &mut HarnessClipboard, preserve_transcript: bool) -> bool {
            self.state.cancelled = true;
            assert!(
                self.state.claim_settlement(),
                "transaction settlement leaked or was claimed twice"
            );
            let can_settle = may_restore(&self.state, self.published_sequence, clipboard.sequence);
            if can_settle {
                let contents = if preserve_transcript {
                    self.transcript.clone()
                } else {
                    self.snapshot.clone()
                };
                clipboard.replace(contents);
            }
            can_settle
        }
    }

    #[test]
    fn windows_stress_harness_runs_thousand_paste_restore_cycles() {
        let mut clipboard = HarnessClipboard::new("initial");

        for cycle in 0..STRESS_CYCLES {
            let original = format!("original-{cycle}");
            let transcript = format!("transcript-{cycle}");
            clipboard.replace(original.clone());

            let mut tx = HarnessTx::publish(&mut clipboard, transcript.clone());
            assert_eq!(
                clipboard.contents, transcript,
                "paste publication was lost on cycle {cycle}"
            );
            assert!(
                tx.settle(&mut clipboard, false),
                "restore ownership lost on cycle {cycle}"
            );
            assert_eq!(
                clipboard.contents, original,
                "clipboard restoration loss on cycle {cycle}"
            );
            assert!(
                tx.state.settled,
                "transaction state leaked on cycle {cycle}"
            );
            assert!(
                !tx.state.claim_settlement(),
                "transaction could be settled twice on cycle {cycle}"
            );

            // Every stress iteration also forces an ownership handoff after a
            // publication. Settlement must detect the newer sequence and leave
            // the newer owner's clipboard untouched.
            clipboard.replace(format!("race-original-{cycle}"));
            let mut raced = HarnessTx::publish(&mut clipboard, format!("race-transcript-{cycle}"));
            let newer = format!("newer-owner-{cycle}");
            clipboard.replace(newer.clone());
            assert!(
                !raced.settle(&mut clipboard, false),
                "ownership clobber was not detected on cycle {cycle}"
            );
            assert_eq!(
                clipboard.contents, newer,
                "newer clipboard owner was clobbered on cycle {cycle}"
            );
            assert!(
                raced.state.settled,
                "raced transaction state leaked on cycle {cycle}"
            );
        }
    }

    #[test]
    fn snapshot_medium_policy_covers_safe_representative_formats() {
        // Clipboard format ids below 0xC000 are fixed Win32 ids. HTML, RTF and
        // custom application formats are registered dynamically in the
        // 0xC000..=0xFFFF range; their storage medium is determined through
        // IDataObject::GetData rather than inferred from the numeric id.
        const CF_HDROP_ID: u32 = 15;
        const REGISTERED_HTML: u32 = 0xC001;
        const REGISTERED_RTF: u32 = 0xC002;
        const REGISTERED_CUSTOM: u32 = 0xC003;

        for format in [
            CF_UNICODETEXT.0 as u32,
            CF_HDROP_ID,
            REGISTERED_HTML,
            REGISTERED_RTF,
            REGISTERED_CUSTOM,
        ] {
            assert_eq!(
                snapshot_medium_kind(format),
                SnapshotMediumKind::HGlobal,
                "safe flat format {format:#x} must be requested as TYMED_HGLOBAL"
            );
        }
        assert_eq!(
            snapshot_medium_kind(CF_BITMAP.0 as u32),
            SnapshotMediumKind::GdiBitmap
        );
        assert_eq!(
            snapshot_medium_kind(CF_ENHMETAFILE.0 as u32),
            SnapshotMediumKind::Skip
        );
        assert_eq!(
            snapshot_medium_kind(CF_PALETTE.0 as u32),
            SnapshotMediumKind::Skip
        );
    }

    #[test]
    fn hglobal_materialization_copies_storage_bytes() {
        let payload = b"Handy clipboard materialization fixture\0";
        unsafe {
            let hg = GlobalAlloc(GMEM_MOVEABLE, payload.len()).expect("allocate test HGLOBAL");
            let ptr = GlobalLock(hg) as *mut u8;
            assert!(!ptr.is_null(), "lock test HGLOBAL");
            std::ptr::copy_nonoverlapping(payload.as_ptr(), ptr, payload.len());
            let _ = GlobalUnlock(hg);

            assert_eq!(
                copy_hglobal_bytes(hg).as_deref(),
                Some(payload.as_slice()),
                "materialization must copy the complete STGMEDIUM HGLOBAL payload"
            );
            let _ = GlobalFree(Some(hg));
        }
    }

    #[test]
    fn normal_transaction_restores_prior_clipboard_unless_preservation_is_requested() {
        let mut clipboard = HarnessClipboard::new("prior clipboard");

        let mut normal = HarnessTx::publish(&mut clipboard, "normal transcript".to_string());
        assert!(normal.settle(&mut clipboard, false));
        assert_eq!(clipboard.contents, "prior clipboard");

        let mut preserve = HarnessTx::publish(&mut clipboard, "kept transcript".to_string());
        assert!(preserve.settle(&mut clipboard, true));
        assert_eq!(clipboard.contents, "kept transcript");

        let mut raced = HarnessTx::publish(&mut clipboard, "raced transcript".to_string());
        clipboard.replace("newer owner");
        assert!(!raced.settle(&mut clipboard, false));
        assert_eq!(clipboard.contents, "newer owner");
    }

    #[test]
    fn normal_receipt_sequenced_transaction_timing_stays_below_regression_budget() {
        const CYCLES: u32 = 20_000;
        const MAX_MEAN_TRANSACTION_TIME: std::time::Duration =
            std::time::Duration::from_micros(1_000);

        // Time the deterministic normal transaction state path end-to-end:
        // snapshot/publish, a legitimate post-injection delayed-render receipt,
        // quiet-period decision, sequence-fenced settlement, and restore. The
        // quiet period is advanced with a synthetic timestamp instead of a sleep,
        // because this guard is for Handy-side regressions rather than target
        // application scheduling latency. Payload values are synthetic fixtures
        // and the report intentionally prints timing metadata only.
        let original = "original clipboard fixture ".repeat(16);
        let transcript = "normal transcript fixture ".repeat(16);
        let mut clipboard = HarnessClipboard::new(original.clone());

        let started = Instant::now();
        for _ in 0..CYCLES {
            clipboard.replace(std::hint::black_box(original.clone()));
            let mut tx =
                HarnessTx::publish(&mut clipboard, std::hint::black_box(transcript.clone()));

            let target_read = Instant::now();
            tx.state.record_receipt(target_read);
            assert!(matches!(
                evaluate(&tx.state, target_read + QUIET_PERIOD),
                WaitDecision::Finish
            ));
            assert!(tx.settle(&mut clipboard, false));
            std::hint::black_box(&clipboard.contents);
        }
        let total = started.elapsed();
        let mean = total / CYCLES;

        println!(
            "windows clipboard normal transaction timing: cycles={CYCLES}, total={total:?}, mean={mean:?}, threshold={MAX_MEAN_TRANSACTION_TIME:?}; clipboard payloads omitted"
        );
        assert!(
            mean <= MAX_MEAN_TRANSACTION_TIME,
            "normal receipt-sequenced transaction mean {mean:?} exceeded {MAX_MEAN_TRANSACTION_TIME:?} regression budget"
        );
    }

    #[test]
    fn format_preservation_bookkeeping_stays_below_normal_paste_regression_budget() {
        const CYCLES: u32 = 20_000;
        const MAX_ADDED_PER_TRANSACTION: std::time::Duration =
            std::time::Duration::from_micros(500);

        let representative = [
            (CF_UNICODETEXT.0 as u32, vec![0_u8; 64]),
            (15_u32, vec![1_u8; 128]),
            (0xC001_u32, vec![2_u8; 256]),
            (0xC002_u32, vec![3_u8; 256]),
            (0xC003_u32, vec![4_u8; 128]),
        ];

        // Baseline models the pre-audit transaction bookkeeping: snapshot one
        // text payload, publish, then restore. The rich path adds representative
        // safe formats and the STGMEDIUM classification. This intentionally does
        // not time SendInput or the receipt quiet-period because those are
        // unchanged by this audit.
        let baseline_start = Instant::now();
        for _ in 0..CYCLES {
            let original = std::hint::black_box(vec![0_u8; 64]);
            let published = std::hint::black_box(vec![9_u8; 64]);
            std::hint::black_box(published);
            std::hint::black_box(original);
        }
        let baseline = baseline_start.elapsed();

        let rich_start = Instant::now();
        for _ in 0..CYCLES {
            let snapshot: Vec<_> = representative
                .iter()
                .filter(|(format, _)| snapshot_medium_kind(*format) == SnapshotMediumKind::HGlobal)
                .map(|(format, data)| (*format, data.clone()))
                .collect();
            let published = std::hint::black_box(vec![9_u8; 64]);
            std::hint::black_box(published);
            std::hint::black_box(snapshot);
        }
        let rich = rich_start.elapsed();
        let added = rich.saturating_sub(baseline) / CYCLES;

        assert!(
            added <= MAX_ADDED_PER_TRANSACTION,
            "format-preservation bookkeeping added {added:?} per transaction; budget is {MAX_ADDED_PER_TRANSACTION:?}"
        );
    }

    #[test]
    fn uipi_failure_preserves_newer_clipboard_owner_and_is_actionable() {
        let mut clipboard = HarnessClipboard::new("original");
        let mut tx = HarnessTx::publish(&mut clipboard, "transcript".to_string());
        tx.state.injection_failed = true;
        assert!(tx.settle(&mut clipboard, false));
        assert_eq!(
            clipboard.contents, "original",
            "UIPI fallback lost the original clipboard"
        );
        assert!(tx.state.settled, "failed transaction state leaked");

        let mut raced = HarnessTx::publish(&mut clipboard, "second-transcript".to_string());
        raced.state.injection_failed = true;
        // Simulate the user/app copying something after Handy published but
        // before the UIPI failure is handled. The newer owner must win.
        clipboard.replace("newer-owner");
        assert!(!raced.settle(&mut clipboard, false));
        assert_eq!(
            clipboard.contents, "newer-owner",
            "newer clipboard owner was clobbered"
        );
        assert!(raced.state.settled, "failed raced transaction state leaked");

        let error = ReliablePasteError::WindowsInputBlocked("SendInput returned no events".into());
        assert!(!error.allows_legacy_fallback());
        let message = error.to_string();
        assert!(message.contains("higher integrity level"));
        assert!(message.contains("same integrity level"));
    }
}
