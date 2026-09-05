//! Receipt-sequenced clipboard paste ("reliable paste", debug-gated).
//!
//! The legacy clipboard paste (`clipboard::paste_via_clipboard`) restores the
//! previous clipboard after a fixed delay. The paste keystroke is only
//! *enqueued* at that point — the target application reads the clipboard
//! whenever its event loop gets to it, so any fixed delay can lose the race
//! and the user gets their old clipboard pasted back (#502).
//!
//! This module instead publishes the transcript as a *lazy promise* and waits
//! for the operating system to tell us that a consumer actually read the
//! clipboard — a "receipt" — before restoring:
//!
//! - Windows: delayed rendering (`SetClipboardData(CF_UNICODETEXT, NULL)`),
//!   the owner window receives `WM_RENDERFORMAT` on read.
//! - macOS: `declareTypes:owner:` with an owner object, the pasteboard calls
//!   `pasteboard:provideDataForType:` on read.
//!
//! Two rules make the receipt trustworthy:
//!
//! 1. Only receipts observed *after* the paste chord was injected count. A
//!    read before that is an eager third party (clipboard manager, antivirus)
//!    reacting to the clipboard change itself.
//! 2. Restoration only happens while we still own the clipboard
//!    (sequence number / changeCount unchanged, no ownership-lost event). If
//!    the user copied something else in the meantime, their action wins.
//!
//! The restore is additionally gated on a short quiet period after the *last*
//! receipt, because some applications read the clipboard several times per
//! paste (Chromium probes, then reads). A bounded timeout caps how long the
//! transcript may occupy the clipboard; the failure mode is always "the
//! transcript stays on the clipboard a bit longer", never "stale content gets
//! pasted".

// The shared transaction state is compiled on all platforms (for the unit
// tests), but only the macOS/Windows platform modules consume all of it.
#![cfg_attr(not(any(target_os = "macos", target_os = "windows")), allow(dead_code))]

use std::fmt;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

/// Opaque foreground target identity used by experimental live insertion.
///
/// The value intentionally contains no window title, document name, or other
/// user-facing media/content metadata. It is only compared for equality inside
/// the current process so a session can fail closed if focus leaves the target
/// application/window captured at recording start.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TargetIdentity(String);

impl TargetIdentity {
    #[cfg(test)]
    pub(crate) fn for_test(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Capture the foreground target without exposing its title or document text.
/// Unsupported/ambiguous environments return `None`; experimental live
/// insertion must treat that as target loss rather than guessing.
pub(crate) fn capture_target_identity() -> Option<TargetIdentity> {
    capture_target_identity_impl()
}

#[cfg(target_os = "windows")]
fn capture_target_identity_impl() -> Option<TargetIdentity> {
    use ::windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    // SAFETY: both functions are read-only foreground-window queries. The HWND
    // is used only as an opaque token and the process id is written to local
    // storage supplied for the documented out parameter.
    unsafe {
        let window = GetForegroundWindow();
        if window.0.is_null() {
            return None;
        }
        let mut process_id = 0_u32;
        let thread_id = GetWindowThreadProcessId(window, Some(&mut process_id));
        if thread_id == 0 || process_id == 0 {
            return None;
        }
        Some(TargetIdentity(format!(
            "windows:{}:{}",
            process_id, window.0 as usize
        )))
    }
}

#[cfg(target_os = "macos")]
fn capture_target_identity_impl() -> Option<TargetIdentity> {
    use objc2_app_kit::NSWorkspace;

    // NSWorkspace exposes the foreground application without Accessibility
    // permission. Using only its pid deliberately avoids collecting app/window
    // names. This still satisfies the hard safety boundary: changing to another
    // foreground application invalidates the session immediately.
    let workspace = NSWorkspace::sharedWorkspace();
    let application = workspace.frontmostApplication()?;
    let process_id = application.processIdentifier();
    (process_id > 0).then(|| TargetIdentity(format!("macos:{process_id}")))
}

#[cfg(target_os = "linux")]
fn capture_target_identity_impl() -> Option<TargetIdentity> {
    use std::process::Command;

    // Cross-application focus inspection is intentionally unavailable on native
    // Wayland. An XWayland window id is not authoritative when a native Wayland
    // surface can become foreground, so fail closed instead of accepting it.
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return None;
    }

    let active = Command::new("xdotool")
        .arg("getactivewindow")
        .output()
        .ok()?;
    if !active.status.success() {
        return None;
    }
    let window = String::from_utf8(active.stdout).ok()?;
    let window = window.trim();
    if window.is_empty() || !window.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    let owner = Command::new("xdotool")
        .args(["getwindowpid", window])
        .output()
        .ok()?;
    if !owner.status.success() {
        return None;
    }
    let process_id = String::from_utf8(owner.stdout).ok()?;
    let process_id = process_id.trim();
    if process_id.is_empty() || !process_id.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    Some(TargetIdentity(format!("x11:{process_id}:{window}")))
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn capture_target_identity_impl() -> Option<TargetIdentity> {
    None
}

/// How long after the *last* observed read the transcript stays on the
/// clipboard before restoring. Covers applications that read the clipboard
/// several times per paste (e.g. Chromium probe-then-read).
pub(crate) const QUIET_PERIOD: Duration = Duration::from_millis(200);

/// Upper bound on how long the transcript may occupy the clipboard before we
/// restore regardless of receipts. Long enough that a realistically loaded
/// target always gets to read first; short enough that a lost keystroke does
/// not strand the transcript on the clipboard for long.
pub(crate) const RESTORE_TIMEOUT: Duration = Duration::from_secs(8);

/// When the chord could not be injected at all, no legitimate receipt can
/// arrive, so restore quickly instead of waiting out the full timeout.
pub(crate) const FAILED_INJECTION_TIMEOUT: Duration = Duration::from_millis(500);

/// Why the receipt-sequenced path could not complete. Most setup failures may
/// safely fall back to the legacy clipboard path. Windows input-injection
/// failures are different: the legacy path uses the same synthetic input APIs,
/// so retrying would obscure the actionable UIPI / integrity-level diagnosis.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReliablePasteError {
    Unavailable(String),
    WindowsInputBlocked(String),
}

impl ReliablePasteError {
    pub(crate) fn allows_legacy_fallback(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }
}

impl From<String> for ReliablePasteError {
    fn from(error: String) -> Self {
        Self::Unavailable(error)
    }
}

impl From<&str> for ReliablePasteError {
    fn from(error: &str) -> Self {
        Self::Unavailable(error.to_string())
    }
}

impl fmt::Display for ReliablePasteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(error) => write!(f, "{error}"),
            Self::WindowsInputBlocked(error) => write!(
                f,
                "Windows did not accept the synthetic paste input ({error}). This can happen when the target runs at a higher integrity level (UIPI). Clipboard ownership was preserved; run Handy at the same integrity level as the target or paste manually."
            ),
        }
    }
}

/// Shared, cross-thread record of one paste transaction.
#[derive(Debug)]
pub(crate) struct TxState {
    /// When the transcript was published to the clipboard.
    pub published_at: Instant,
    /// When the paste chord was injected. Only receipts *after* this count as
    /// evidence the target read the transcript — earlier reads are eager third
    /// parties reacting to the clipboard change itself.
    pub injected_at: Option<Instant>,
    /// The chord could not be sent; short-circuit the wait.
    pub injection_failed: bool,
    /// Times at which a consumer requested the clipboard data.
    pub receipts: Vec<Instant>,
    /// Someone else took clipboard ownership (user copied elsewhere, ...).
    pub ownership_lost: bool,
    /// A newer paste transaction settled this one early (see flush logic in
    /// the platform modules).
    pub cancelled: bool,
    /// Exactly one thread may claim settlement. This prevents a timer tick and
    /// a synchronous failure/flush path from both restoring the clipboard.
    pub settled: bool,
    /// The post-paste Enter (auto-submit) has been sent for this transaction.
    /// (Read on Windows; the macOS path settles via `MacPending::settled`.)
    #[allow(dead_code)]
    pub auto_submit_sent: bool,
    /// First post-injection receipt has been logged.
    pub logged_receipt: bool,
}

impl TxState {
    pub fn new() -> Self {
        Self {
            published_at: Instant::now(),
            injected_at: None,
            injection_failed: false,
            receipts: Vec::new(),
            ownership_lost: false,
            cancelled: false,
            settled: false,
            auto_submit_sent: false,
            logged_receipt: false,
        }
    }

    /// Records a read receipt, logging the first one that counts as evidence.
    pub fn record_receipt(&mut self, at: Instant) {
        self.receipts.push(at);
        if !self.logged_receipt {
            if let Some(injected) = self.injected_at {
                if at >= injected {
                    self.logged_receipt = true;
                    log::info!(
                        "[reliable-paste] clipboard read {}ms after chord",
                        at.duration_since(injected).as_millis()
                    );
                }
            }
        }
    }

    pub fn last_receipt_after_injection(&self) -> Option<Instant> {
        let injected = self.injected_at?;
        self.receipts.iter().copied().rev().find(|t| *t >= injected)
    }

    pub fn any_receipt_after_injection(&self) -> bool {
        self.last_receipt_after_injection().is_some()
    }

    /// Claims responsibility for settling this transaction. A second caller
    /// loses the claim and must not touch the clipboard.
    pub fn claim_settlement(&mut self) -> bool {
        if self.settled {
            false
        } else {
            self.settled = true;
            true
        }
    }
}

pub(crate) enum WaitDecision {
    KeepWaiting,
    /// Stop waiting; settle the transaction (auto-submit + guarded restore).
    Finish,
}

/// Restoration is permitted only while the clipboard still represents this
/// transaction. Cancellation itself does not forbid restoration — a newer Handy
/// transaction deliberately cancels and settles the previous one — but any
/// ownership-loss event or sequence change means a newer owner wins.
pub(crate) fn may_restore(state: &TxState, published_sequence: u32, current_sequence: u32) -> bool {
    !state.ownership_lost && published_sequence == current_sequence
}

/// Pure decision: given the current transaction state, keep waiting for the
/// target to read, or finish now. Both platform event loops call this.
pub(crate) fn evaluate(state: &TxState, now: Instant) -> WaitDecision {
    if state.ownership_lost || state.cancelled || state.settled {
        return WaitDecision::Finish;
    }
    if let Some(last) = state.last_receipt_after_injection() {
        if now.duration_since(last) >= QUIET_PERIOD {
            return WaitDecision::Finish;
        }
    }
    let deadline = if state.injection_failed {
        FAILED_INJECTION_TIMEOUT
    } else {
        RESTORE_TIMEOUT
    };
    if now.duration_since(state.published_at) >= deadline {
        return WaitDecision::Finish;
    }
    WaitDecision::KeepWaiting
}

/// Modifier hold for the receipt-sequenced chord, kept at parity with the
/// legacy path (100ms) for the beta: that hold was added in #165 because real
/// users' systems dropped chords released too quickly, and the beta should
/// validate the receipt mechanism without changing a second variable.
///
/// Once receipts are proven in the field this becomes a safe tuning knob — a
/// chord the target never recognizes produces no receipt and is logged ("no
/// read within timeout") rather than failing silently, so a shorter hold
/// (measured working at 10ms on a fast machine, cutting visible latency from
/// ~110ms to ~20ms) can be tried as its own experiment later.
const CHORD_HOLD_MS: u64 = 100;

/// Sends the platform paste chord for the configured method.
pub(crate) fn send_chord(
    enigo: &mut enigo::Enigo,
    paste_method: &crate::settings::PasteMethod,
) -> Result<(), String> {
    use crate::settings::PasteMethod;
    match paste_method {
        PasteMethod::CtrlV => crate::input::send_paste_ctrl_v(enigo, CHORD_HOLD_MS),
        PasteMethod::CtrlShiftV => crate::input::send_paste_ctrl_shift_v(enigo, CHORD_HOLD_MS),
        PasteMethod::ShiftInsert => crate::input::send_paste_shift_insert(enigo, CHORD_HOLD_MS),
        other => Err(format!(
            "Invalid paste method for clipboard paste: {:?}",
            other
        )),
    }
}

/// Attempts the receipt-sequenced paste. Returns `Err` before anything has
/// been published when the platform transaction cannot start, in which case
/// the caller should fall back to the legacy paste path. On `Ok`, publishing
/// and chord injection have completed and the guarded restore (plus
/// auto-submit) finishes asynchronously.
#[cfg(target_os = "windows")]
pub(crate) fn shutdown_pending() {
    windows::shutdown_pending();
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn try_reliable_paste(
    text: &str,
    app_handle: &tauri::AppHandle,
    paste_method: &crate::settings::PasteMethod,
    enigo: &mut enigo::Enigo,
    auto_submit: bool,
    auto_submit_key: crate::settings::AutoSubmitKey,
    clipboard_handling: crate::settings::ClipboardHandling,
) -> Result<(), ReliablePasteError> {
    #[cfg(target_os = "macos")]
    {
        macos::run(
            text,
            app_handle,
            paste_method,
            enigo,
            auto_submit,
            auto_submit_key,
            clipboard_handling,
        )
        .map_err(ReliablePasteError::Unavailable)
    }

    #[cfg(target_os = "windows")]
    {
        windows::run(
            text,
            app_handle,
            paste_method,
            enigo,
            auto_submit,
            auto_submit_key,
            clipboard_handling,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_after_publish(published_ago: Duration) -> TxState {
        let mut s = TxState::new();
        s.published_at = Instant::now() - published_ago;
        s
    }

    #[test]
    fn keeps_waiting_without_receipt_within_timeout() {
        let s = state_after_publish(Duration::from_millis(100));
        assert!(matches!(
            evaluate(&s, Instant::now()),
            WaitDecision::KeepWaiting
        ));
    }

    #[test]
    fn finishes_after_quiet_period_once_read() {
        let mut s = state_after_publish(Duration::from_millis(300));
        s.injected_at = Some(Instant::now() - Duration::from_millis(250));
        s.receipts.push(Instant::now() - QUIET_PERIOD);
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn waits_through_quiet_period_after_recent_read() {
        let mut s = state_after_publish(Duration::from_millis(300));
        s.injected_at = Some(Instant::now() - Duration::from_millis(100));
        s.receipts.push(Instant::now() - Duration::from_millis(50));
        assert!(matches!(
            evaluate(&s, Instant::now()),
            WaitDecision::KeepWaiting
        ));
    }

    #[test]
    fn pre_injection_receipt_does_not_count() {
        let mut s = state_after_publish(Duration::from_millis(300));
        s.receipts.push(Instant::now() - Duration::from_millis(200));
        s.injected_at = Some(Instant::now() - Duration::from_millis(100));
        assert!(!s.any_receipt_after_injection());
        assert!(matches!(
            evaluate(&s, Instant::now()),
            WaitDecision::KeepWaiting
        ));
    }

    #[test]
    fn finishes_on_timeout_without_receipt() {
        let s = state_after_publish(RESTORE_TIMEOUT);
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn failed_injection_uses_short_timeout() {
        let mut s = state_after_publish(FAILED_INJECTION_TIMEOUT);
        s.injection_failed = true;
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn ownership_loss_finishes_immediately() {
        let mut s = state_after_publish(Duration::from_millis(10));
        s.ownership_lost = true;
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn app_exit_or_explicit_cancellation_finishes_immediately() {
        let mut s = state_after_publish(Duration::from_millis(10));
        s.cancelled = true;
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn settlement_can_only_be_claimed_once() {
        let mut s = state_after_publish(Duration::from_millis(10));
        assert!(s.claim_settlement());
        assert!(!s.claim_settlement());
        assert!(matches!(evaluate(&s, Instant::now()), WaitDecision::Finish));
    }

    #[test]
    fn newer_clipboard_sequence_always_wins() {
        let s = state_after_publish(Duration::from_millis(10));
        assert!(may_restore(&s, 41, 41));
        assert!(!may_restore(&s, 41, 42));

        let mut ownership_lost = s;
        ownership_lost.ownership_lost = true;
        assert!(!may_restore(&ownership_lost, 41, 41));
    }

    #[test]
    fn delayed_render_and_clipboard_manager_races_survive_thousand_cycles() {
        for _ in 0..1_000 {
            let now = Instant::now();
            let mut s = TxState::new();
            s.published_at = now - Duration::from_millis(500);

            // An eager clipboard manager can request the delayed format as soon
            // as it is advertised. That pre-injection receipt must not settle
            // the transaction or authorize auto-submit.
            s.record_receipt(now - Duration::from_millis(300));
            s.injected_at = Some(now - Duration::from_millis(200));
            assert!(!s.any_receipt_after_injection());
            assert!(matches!(evaluate(&s, now), WaitDecision::KeepWaiting));

            // The target's delayed-render read after the paste chord is the real
            // receipt. Keep waiting through the quiet period, then settle.
            let target_read = now - Duration::from_millis(100);
            s.record_receipt(target_read);
            assert!(s.any_receipt_after_injection());
            assert!(matches!(
                evaluate(&s, target_read + QUIET_PERIOD - Duration::from_millis(1)),
                WaitDecision::KeepWaiting
            ));
            assert!(matches!(
                evaluate(&s, target_read + QUIET_PERIOD),
                WaitDecision::Finish
            ));
            assert!(may_restore(&s, 7, 7));
            assert!(!may_restore(&s, 7, 8));
        }
    }
}
