use crate::actions::ACTION_MAP;
use crate::managers::audio::AudioRecordingManager;
use crate::settings::{InsertionMode, ShortcutActivation};
use log::{debug, error, warn};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

const DEBOUNCE: Duration = Duration::from_millis(30);
const RELEASE_GRACE: Duration = Duration::from_millis(50);

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveInsertionAttemptKind {
    CommittedDelta,
    FinalTail,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveInsertionOutcome {
    Inserted,
    TargetLost,
    TargetChanged,
    InputFailed,
    ClipboardOwnershipLost,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveInsertionStopReason {
    TargetLost,
    TargetChanged,
    InputFailed,
    ClipboardOwnershipLost,
    RevisedInsertedPrefix,
    Cancelled,
}

/// Text returned to the input adapter for one append-only insertion attempt.
/// The sequence is required when recording the result so a stale asynchronous
/// completion cannot advance a newer attempt's ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiveInsertionAttempt {
    pub sequence: u64,
    pub kind: LiveInsertionAttemptKind,
    pub text: String,
    /// Expected foreground target. The input adapter must re-check this identity
    /// immediately before injection; the ledger's planning-time check alone is
    /// not sufficient because focus can change while an async attempt is queued.
    pub target: crate::paste_tx::TargetIdentity,
}

/// Metadata retained after an attempt. Text itself is intentionally not copied
/// into the result ledger: the authoritative inserted/pending strings already
/// exist on the session and diagnostics only need sequence/kind/size/outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiveInsertionAttemptRecord {
    pub sequence: u64,
    pub kind: LiveInsertionAttemptKind,
    pub byte_len: usize,
    pub outcome: Option<LiveInsertionOutcome>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PendingLiveInsertion {
    attempt: LiveInsertionAttempt,
    base_inserted_len: usize,
}

/// Per-recording append-only ledger for committed streaming text.
///
/// The ledger never edits text already sent to another application. A new model
/// hypothesis must start with the exact prefix that was confirmed inserted; if
/// it revises that prefix, live insertion stops for the rest of the session.
/// This raw-prefix rule is the append-safe deterministic transform: whole-text
/// cleanup (especially AI cleanup) must happen only in preview/final-only modes,
/// never after live deltas have escaped Handy.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct LiveInsertionLedger {
    mode: InsertionMode,
    target: Option<crate::paste_tx::TargetIdentity>,
    inserted_committed: String,
    latest_committed: String,
    pending_text: String,
    attempts: Vec<LiveInsertionAttemptRecord>,
    safety_events: Vec<LiveInsertionStopReason>,
    pending_attempt: Option<PendingLiveInsertion>,
    speech_evidence: bool,
    cancelled: bool,
    finalization_requested: bool,
    final_tail_flushed: bool,
    stop_reason: Option<LiveInsertionStopReason>,
    next_sequence: u64,
}

#[allow(dead_code)]
impl LiveInsertionLedger {
    pub(crate) fn new(
        mode: InsertionMode,
        target: Option<crate::paste_tx::TargetIdentity>,
    ) -> Self {
        Self {
            mode,
            target,
            inserted_committed: String::new(),
            latest_committed: String::new(),
            pending_text: String::new(),
            attempts: Vec::new(),
            safety_events: Vec::new(),
            pending_attempt: None,
            speech_evidence: false,
            cancelled: false,
            finalization_requested: false,
            final_tail_flushed: false,
            stop_reason: None,
            next_sequence: 1,
        }
    }

    pub(crate) fn mode(&self) -> InsertionMode {
        self.mode
    }

    pub(crate) fn inserted_committed(&self) -> &str {
        &self.inserted_committed
    }

    pub(crate) fn pending_text(&self) -> &str {
        &self.pending_text
    }

    pub(crate) fn attempts(&self) -> &[LiveInsertionAttemptRecord] {
        &self.attempts
    }

    pub(crate) fn safety_events(&self) -> &[LiveInsertionStopReason] {
        &self.safety_events
    }

    pub(crate) fn stop_reason(&self) -> Option<LiveInsertionStopReason> {
        self.stop_reason
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub(crate) fn final_tail_flushed(&self) -> bool {
        self.final_tail_flushed
    }

    fn is_live(&self) -> bool {
        self.mode == InsertionMode::LiveCommittedExperimental
    }

    fn stop(&mut self, reason: LiveInsertionStopReason) {
        if self.stop_reason.is_none() {
            self.stop_reason = Some(reason);
            self.safety_events.push(reason);
        }
        self.pending_text.clear();
    }

    fn target_is_current(
        &mut self,
        current_target: Option<&crate::paste_tx::TargetIdentity>,
    ) -> bool {
        let result = match (self.target.as_ref(), current_target) {
            (Some(expected), Some(current)) if expected == current => Ok(()),
            (Some(_), Some(_)) => Err(LiveInsertionStopReason::TargetChanged),
            _ => Err(LiveInsertionStopReason::TargetLost),
        };
        match result {
            Ok(()) => true,
            Err(reason) => {
                self.stop(reason);
                false
            }
        }
    }

    /// Exact-prefix suffix extraction is deliberately the only transform used by
    /// live mode. It cannot rewrite or delete text that has already escaped.
    fn append_safe_suffix<'a>(&self, candidate: &'a str) -> Option<&'a str> {
        candidate.strip_prefix(&self.inserted_committed)
    }

    fn refresh_pending_text(&mut self) -> bool {
        match self.append_safe_suffix(&self.latest_committed) {
            Some(suffix) => {
                self.pending_text = suffix.to_string();
                true
            }
            None => {
                self.stop(LiveInsertionStopReason::RevisedInsertedPrefix);
                false
            }
        }
    }

    fn plan_latest(
        &mut self,
        kind: LiveInsertionAttemptKind,
        current_target: Option<&crate::paste_tx::TargetIdentity>,
    ) -> Option<LiveInsertionAttempt> {
        if !self.is_live()
            || self.cancelled
            || self.stop_reason.is_some()
            || !self.speech_evidence
            || self.pending_attempt.is_some()
        {
            return None;
        }
        if !self.target_is_current(current_target) || !self.refresh_pending_text() {
            return None;
        }
        if self.pending_text.is_empty() {
            return None;
        }

        let attempt = LiveInsertionAttempt {
            sequence: self.next_sequence,
            kind,
            text: self.pending_text.clone(),
            target: self
                .target
                .clone()
                .expect("live planning requires a captured target"),
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.attempts.push(LiveInsertionAttemptRecord {
            sequence: attempt.sequence,
            kind,
            byte_len: attempt.text.len(),
            outcome: None,
        });
        self.pending_attempt = Some(PendingLiveInsertion {
            base_inserted_len: self.inserted_committed.len(),
            attempt: attempt.clone(),
        });
        Some(attempt)
    }

    /// Observe a streaming model's committed prefix. Tentative text is not an
    /// argument to this API, so it cannot accidentally reach the insertion path.
    pub(crate) fn observe_committed(
        &mut self,
        committed: &str,
        current_target: Option<&crate::paste_tx::TargetIdentity>,
    ) -> Option<LiveInsertionAttempt> {
        self.latest_committed.clear();
        self.latest_committed.push_str(committed);

        if !self.is_live() || self.cancelled || self.stop_reason.is_some() {
            return None;
        }
        if !self.refresh_pending_text() {
            return None;
        }
        self.plan_latest(LiveInsertionAttemptKind::CommittedDelta, current_target)
    }

    /// Latch positive speech evidence for the session. A committed-looking model
    /// emission observed before this call remains pending and becomes eligible
    /// only after this positive latch and a successful focus check.
    pub(crate) fn observe_speech_evidence(
        &mut self,
        current_target: Option<&crate::paste_tx::TargetIdentity>,
    ) -> Option<LiveInsertionAttempt> {
        self.speech_evidence = true;
        self.plan_latest(LiveInsertionAttemptKind::CommittedDelta, current_target)
    }

    /// Record the result of the exact attempt returned by this ledger. Unknown,
    /// duplicate, or stale sequence ids are ignored. A failed input attempt is
    /// terminal because partial delivery cannot be ruled out safely; retrying
    /// could duplicate text. Clipboard ownership loss is terminal for the same
    /// reason and leaves the user's newer clipboard owner untouched.
    pub(crate) fn record_attempt_result(
        &mut self,
        sequence: u64,
        outcome: LiveInsertionOutcome,
    ) -> bool {
        let Some(pending) = self.pending_attempt.as_ref() else {
            return false;
        };
        if pending.attempt.sequence != sequence {
            return false;
        }

        let Some(record) = self
            .attempts
            .iter_mut()
            .find(|record| record.sequence == sequence)
        else {
            return false;
        };
        if record.outcome.is_some() {
            return false;
        }
        record.outcome = Some(outcome);

        let pending = self.pending_attempt.take().expect("checked above");
        match outcome {
            LiveInsertionOutcome::Inserted => {
                // Only the attempt created from the current confirmed prefix may
                // advance it. A stale completion cannot overwrite newer state.
                if self.inserted_committed.len() != pending.base_inserted_len {
                    self.stop(LiveInsertionStopReason::RevisedInsertedPrefix);
                    return false;
                }
                self.inserted_committed.push_str(&pending.attempt.text);
                if pending.attempt.kind == LiveInsertionAttemptKind::FinalTail {
                    self.final_tail_flushed = true;
                }
                // A newer stream event may have arrived while input was in
                // flight. If it revised the just-inserted text, stop now rather
                // than attempting generic deletion/replacement. Cancellation is
                // already terminal, so keep its pending text empty instead.
                if self.cancelled {
                    self.pending_text.clear();
                } else {
                    let _ = self.refresh_pending_text();
                }
            }
            LiveInsertionOutcome::TargetLost => {
                self.stop(LiveInsertionStopReason::TargetLost);
            }
            LiveInsertionOutcome::TargetChanged => {
                self.stop(LiveInsertionStopReason::TargetChanged);
            }
            LiveInsertionOutcome::InputFailed => {
                self.stop(LiveInsertionStopReason::InputFailed);
            }
            LiveInsertionOutcome::ClipboardOwnershipLost => {
                self.stop(LiveInsertionStopReason::ClipboardOwnershipLost);
            }
        }
        true
    }

    /// Stop accepting new live insertions. An already in-flight result may still
    /// be recorded so the ledger reflects what could have reached the target,
    /// but cancellation never schedules a final tail.
    pub(crate) fn cancel(&mut self) {
        if self.cancelled {
            return;
        }
        self.cancelled = true;
        self.stop(LiveInsertionStopReason::Cancelled);
    }

    /// Plan the remaining raw tail at stop. This is at most one append-only
    /// attempt. If the final model text revised any prefix already inserted, the
    /// ledger records the limitation and inserts nothing rather than trying to
    /// repair another application's document generically.
    pub(crate) fn finalize(
        &mut self,
        final_text: &str,
        current_target: Option<&crate::paste_tx::TargetIdentity>,
    ) -> Option<LiveInsertionAttempt> {
        if !self.is_live()
            || self.cancelled
            || self.stop_reason.is_some()
            || self.finalization_requested
        {
            return None;
        }
        // A caller using asynchronous input must settle the preceding attempt
        // before finalizing. Returning None without marking finalization lets it
        // retry after that exact completion arrives.
        if self.pending_attempt.is_some() {
            return None;
        }

        self.latest_committed.clear();
        self.latest_committed.push_str(final_text);
        self.finalization_requested = true;

        if !self.speech_evidence {
            self.pending_text.clear();
            return None;
        }
        if !self.target_is_current(current_target) || !self.refresh_pending_text() {
            return None;
        }
        if self.pending_text.is_empty() {
            self.final_tail_flushed = true;
            return None;
        }
        self.plan_latest(LiveInsertionAttemptKind::FinalTail, current_target)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PttAction {
    Passthrough,
    DeferRelease,
    CancelRelease,
}

/// A key-up deferred by `RELEASE_GRACE` so a synthesized X11 auto-repeat
/// press can cancel it (#1539). When the grace elapses the hold is resolved
/// by [`CoordinatorState::finish_hold`] (recording) or
/// [`CoordinatorState::finish_pending_hold`] (press remembered while busy).
struct PendingRelease {
    binding_id: String,
    hotkey_string: String,
    deadline: Instant,
    /// When the key actually went up. The hold duration is measured to this
    /// instant, not to the grace expiry.
    released_at: Instant,
    /// Holds at least this long stop recording; shorter ones lock it on.
    /// Push-to-talk passes zero so every release stops.
    hold_threshold: Duration,
}

/// A press that arrived while the pipeline was still busy processing the
/// previous transcription. Toggle-style triggers (SIGUSR2, CLI flags, some
/// pedal setups) flip state on every edge, so dropping a busy press desyncs
/// the parity: the next edge starts a recording nobody will ever stop.
struct PendingPress {
    binding_id: String,
    hotkey_string: String,
    /// The real key-down time, so a hold that straddles the drain is still
    /// measured from when the user pressed, not from when recording began.
    pressed_at: Instant,
    /// The recording will start locked on when the pipeline drains: set from
    /// the start for toggle, and for hold-or-toggle once the key came back up
    /// within the threshold (a tap). An unlocked pending press is a key we
    /// believe is still held.
    locked: bool,
}

impl PendingPress {
    fn remembered(&self) -> Remembered {
        if self.locked {
            Remembered::Locked
        } else {
            Remembered::Held
        }
    }
}

/// What kind of press is already waiting for the pipeline to drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Remembered {
    /// The key is still down as far as we know.
    Held,
    /// A toggle press or a classified tap: it will start a locked session.
    Locked,
}

/// Bookkeeping for the key press that started the current recording.
struct Hold {
    pressed_at: Instant,
    /// Recording outlives the key: the next press stops it, releases are
    /// ignored. Always set for toggle; set for hold-or-toggle once a release
    /// has been classified as a tap.
    locked: bool,
}

/// What to do with an input that arrives while the pipeline is busy
/// (`Stage::Processing`). `remembered` is the press for the same binding
/// already waiting for the pipeline to drain, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BusyAction {
    /// Ignore the input entirely.
    Ignore,
    /// Remember the press; start recording when the pipeline finishes.
    Remember,
    /// This press cancels a previously remembered press: two presses during
    /// one busy window net to no-op, exactly as a press stops a locked
    /// session once recording.
    Forget,
}

fn classify_busy_input(
    is_pressed: bool,
    mode: ShortcutActivation,
    remembered: Option<Remembered>,
) -> BusyAction {
    use ShortcutActivation::*;
    match (mode, is_pressed, remembered) {
        // Toggle: presses alternate remember/forget to preserve parity.
        (Toggle, true, Some(_)) => BusyAction::Forget,
        (Toggle, true, None) => BusyAction::Remember,
        // Toggle mode ignores releases.
        (Toggle, false, _) => BusyAction::Ignore,
        // Hold modes: a press while busy means the user is holding the key —
        // start as soon as the pipeline drains. A press on a queued tap stops
        // it (parity); a press while the key is already down is a repeat.
        (PushToTalk | HoldOrToggle, true, None) => BusyAction::Remember,
        (PushToTalk | HoldOrToggle, true, Some(Remembered::Locked)) => BusyAction::Forget,
        (PushToTalk | HoldOrToggle, true, Some(Remembered::Held)) => BusyAction::Ignore,
        // Releases of a held pending press are deferred by the grace window
        // before reaching here and resolved by `finish_pending_hold`; any
        // other release (no press remembered, or already locked) is noise.
        (PushToTalk | HoldOrToggle, false, _) => BusyAction::Ignore,
    }
}

/// Pipeline lifecycle.
#[derive(Debug, PartialEq, Eq)]
enum Stage {
    Idle,
    Recording(String), // binding_id
    Processing,
}

/// A keyboard/signal edge for a transcribe binding.
struct InputEvent {
    binding_id: String,
    hotkey_string: String,
    is_pressed: bool,
    mode: ShortcutActivation,
    /// Hold-or-toggle: minimum press duration that counts as a hold.
    hold_threshold: Duration,
    /// External triggers (SIGUSR2, CLI flags) rather than physical keys.
    /// They fire on every edge by design and must never be debounced —
    /// dropping one desyncs toggle parity and wedges recording on.
    external: bool,
}

impl InputEvent {
    /// The hold duration at or above which a release stops recording.
    fn effective_hold_threshold(&self) -> Duration {
        match self.mode {
            ShortcutActivation::HoldOrToggle => self.hold_threshold,
            // Every release stops; toggle never defers releases at all.
            ShortcutActivation::PushToTalk | ShortcutActivation::Toggle => Duration::ZERO,
        }
    }
}

/// A side effect decided by [`CoordinatorState`]; the coordinator thread is
/// the only executor. Keeping decisions pure lets tests drive the exact
/// production transitions without a Tauri `AppHandle` or real timers.
#[derive(Debug, PartialEq, Eq)]
enum Effect {
    Start {
        binding_id: String,
        hotkey_string: String,
    },
    Stop {
        binding_id: String,
        hotkey_string: String,
    },
}

/// Commands processed sequentially by the coordinator thread.
enum Command {
    Input(InputEvent),
    Cancel { recording_was_active: bool },
    ProcessingFinished,
}

/// Decide whether a key-up should be deferred (so auto-repeat can cancel it)
/// or a key-down cancels a deferred release. `hold_to_talk` is whether a
/// release currently ends the session: true for push-to-talk and for an
/// unlocked hold-or-toggle session, false for toggle and a locked session.
/// `held_binding` is the binding whose key we believe is down — the one
/// recording, or the one remembered while the pipeline is busy.
fn classify_ptt_event(
    pending_release_binding: Option<&str>,
    is_pressed: bool,
    hold_to_talk: bool,
    binding_id: &str,
    held_binding: Option<&str>,
) -> PttAction {
    if !hold_to_talk {
        return PttAction::Passthrough;
    }

    if is_pressed {
        if pending_release_binding == Some(binding_id) {
            PttAction::CancelRelease
        } else {
            PttAction::Passthrough
        }
    } else if held_binding == Some(binding_id) && pending_release_binding.is_none() {
        PttAction::DeferRelease
    } else {
        PttAction::Passthrough
    }
}

/// Pure lifecycle state machine: owns every transition decision (release
/// grace, hold-vs-tap classification, debounce, busy-pipeline
/// remember/forget, cancel, drain). Produces [`Effect`]s instead of touching
/// the app, so unit tests exercise the real production logic.
///
/// All three activation modes run through one machine. A recording starts on
/// key-down in every mode; what differs is how it ends:
///
/// * push-to-talk — every release stops (hold threshold of zero)
/// * toggle — releases are ignored, the next press stops (locked from the start)
/// * hold-or-toggle — a release after a long hold stops; a release after a
///   short tap locks the session, and the next press stops
struct CoordinatorState {
    stage: Stage,
    hold: Option<Hold>,
    last_press: Option<Instant>,
    pending_release: Option<PendingRelease>,
    pending_press: Option<PendingPress>,
}

impl CoordinatorState {
    fn new() -> Self {
        Self {
            stage: Stage::Idle,
            hold: None,
            last_press: None,
            pending_release: None,
            pending_press: None,
        }
    }

    /// Deadline of the deferred release, if any — drives `recv_timeout`.
    fn grace_deadline(&self) -> Option<Instant> {
        self.pending_release.as_ref().map(|p| p.deadline)
    }

    /// Whether the current session (recording, or remembered for the drain)
    /// outlives the key, so releases are ignored and the next press ends it.
    fn is_locked(&self) -> bool {
        self.hold.as_ref().is_some_and(|h| h.locked)
            || self.pending_press.as_ref().is_some_and(|p| p.locked)
    }

    fn on_input(&mut self, input: InputEvent, now: Instant) -> Option<Effect> {
        let pending_release_binding = self
            .pending_release
            .as_ref()
            .map(|pending| pending.binding_id.as_str());
        let held_binding = match &self.stage {
            Stage::Recording(id) => Some(id.as_str()),
            Stage::Processing => self.pending_press.as_ref().map(|p| p.binding_id.as_str()),
            Stage::Idle => None,
        };
        let hold_to_talk = input.mode != ShortcutActivation::Toggle && !self.is_locked();

        match classify_ptt_event(
            pending_release_binding,
            input.is_pressed,
            hold_to_talk,
            &input.binding_id,
            held_binding,
        ) {
            PttAction::CancelRelease => {
                self.pending_release = None;
                return None;
            }
            PttAction::DeferRelease => {
                self.pending_release = Some(PendingRelease {
                    hold_threshold: input.effective_hold_threshold(),
                    binding_id: input.binding_id,
                    hotkey_string: input.hotkey_string,
                    deadline: now + RELEASE_GRACE,
                    released_at: now,
                });
                return None;
            }
            PttAction::Passthrough => {}
        }

        // Debounce rapid-fire press events (key repeat / double-tap).
        // Releases in the hold modes are deferred above to absorb X11 auto-repeat.
        // External triggers are exempt: each one is a deliberate edge from the
        // user's own integration, and dropping it desyncs toggle parity.
        if input.is_pressed && !input.external {
            if self
                .last_press
                .is_some_and(|t| now.duration_since(t) < DEBOUNCE)
            {
                debug!("Debounced press for '{}'", input.binding_id);
                return None;
            }
            self.last_press = Some(now);
        }

        // A busy pipeline can't accept lifecycle changes now: classify the
        // input against any already-remembered press instead of dropping it
        // silently.
        if let Stage::Processing = self.stage {
            // Only one press can be remembered. Once a binding has claimed it,
            // inputs for a different binding are ignored — the same rule as a
            // different binding pressed while recording — rather than silently
            // replacing the remembered press and breaking its parity.
            if let Some(pending) = &self.pending_press {
                if pending.binding_id != input.binding_id {
                    debug!(
                        "Ignoring input for '{}': '{}' is already pending",
                        input.binding_id, pending.binding_id
                    );
                    return None;
                }
            }
            let remembered = self.pending_press.as_ref().map(|p| p.remembered());
            match classify_busy_input(input.is_pressed, input.mode, remembered) {
                BusyAction::Remember => {
                    debug!(
                        "Remembering press for '{}': pipeline busy",
                        input.binding_id
                    );
                    self.pending_press = Some(PendingPress {
                        // Toggle never ends on a release: locked from the start.
                        locked: input.mode == ShortcutActivation::Toggle,
                        binding_id: input.binding_id,
                        hotkey_string: input.hotkey_string,
                        pressed_at: now,
                    });
                }
                BusyAction::Forget => {
                    debug!("Forgetting remembered press for '{}'", input.binding_id);
                    self.pending_press = None;
                }
                BusyAction::Ignore => {
                    debug!("Ignoring input for '{}': pipeline busy", input.binding_id);
                }
            }
            return None;
        }

        if input.is_pressed {
            match &self.stage {
                Stage::Idle => {
                    // Toggle never ends on a release: locked from the start.
                    let locked = input.mode == ShortcutActivation::Toggle;
                    return Some(self.begin_recording(
                        input.binding_id,
                        input.hotkey_string,
                        now,
                        locked,
                    ));
                }
                Stage::Recording(id) if id == &input.binding_id => {
                    // A locked session ends on the next press. In toggle mode
                    // every press ends it, even if the recording began under a
                    // hold mode (the setting changed mid-recording) — otherwise
                    // nothing but Escape could stop it.
                    if self.is_locked() || input.mode == ShortcutActivation::Toggle {
                        return Some(self.begin_processing(input.binding_id, input.hotkey_string));
                    }
                    // The key is still held (its release will end this
                    // recording), so a repeated press means nothing.
                    debug!("Ignoring press for '{}': key is held", input.binding_id);
                }
                _ => debug!(
                    "Ignoring press for '{}': another binding is recording",
                    input.binding_id
                ),
            }
        } else if hold_to_talk
            && matches!(&self.stage, Stage::Recording(id) if id == &input.binding_id)
        {
            // A release that was not deferred (one is already pending for this
            // binding): resolve it immediately rather than dropping it.
            let threshold = input.effective_hold_threshold();
            return self.finish_hold(input.binding_id, input.hotkey_string, now, threshold);
        }
        None
    }

    /// The `RELEASE_GRACE` window elapsed with no cancelling press arriving:
    /// resolve the deferred release against whatever that binding's key was
    /// holding — the live recording, or a press remembered while busy.
    fn on_grace_expired(&mut self) -> Option<Effect> {
        let pending = self.pending_release.take()?;
        match &self.stage {
            Stage::Recording(id) if *id == pending.binding_id => self.finish_hold(
                pending.binding_id,
                pending.hotkey_string,
                pending.released_at,
                pending.hold_threshold,
            ),
            Stage::Processing => {
                self.finish_pending_hold(&pending);
                None
            }
            _ => None,
        }
    }

    /// A press remembered while the pipeline was busy has been released for
    /// real, still before the drain. A completed hold has nothing left to
    /// start; a tap queues a locked session so the drain starts it — the
    /// same hold-vs-tap rule as [`CoordinatorState::finish_hold`].
    fn finish_pending_hold(&mut self, release: &PendingRelease) {
        let Some(pending) = self
            .pending_press
            .as_mut()
            .filter(|p| p.binding_id == release.binding_id)
        else {
            return;
        };
        let held = release
            .released_at
            .saturating_duration_since(pending.pressed_at);
        if held < release.hold_threshold {
            debug!(
                "Tap ({held:?}) for '{}' while busy: will start locked on when the pipeline drains",
                release.binding_id
            );
            pending.locked = true;
        } else {
            debug!(
                "Forgetting remembered press for '{}': released after a {held:?} hold while busy",
                release.binding_id
            );
            self.pending_press = None;
        }
    }

    /// The key that started the current recording has been released for real.
    /// A hold at least `threshold` long stops recording; anything shorter was a
    /// tap, which locks the session on until the next press.
    fn finish_hold(
        &mut self,
        binding_id: String,
        hotkey_string: String,
        released_at: Instant,
        threshold: Duration,
    ) -> Option<Effect> {
        let held = self
            .hold
            .as_ref()
            .map(|h| released_at.saturating_duration_since(h.pressed_at))
            // No hold bookkeeping means we cannot tell a tap from a hold;
            // stopping is the safe reading (it is what push-to-talk always did).
            .unwrap_or(Duration::MAX);
        if held >= threshold {
            return Some(self.begin_processing(binding_id, hotkey_string));
        }
        if let Some(hold) = &mut self.hold {
            debug!("Tap ({held:?}) for '{binding_id}': recording locked on until the next press");
            hold.locked = true;
        }
        None
    }

    fn on_cancel(&mut self, recording_was_active: bool) {
        self.pending_release = None;
        // An explicit cancel abandons any remembered start too — the user
        // asked for silence, not a deferred recording.
        self.pending_press = None;
        // Don't reset during processing — wait for the pipeline to finish.
        if !matches!(self.stage, Stage::Processing)
            && (recording_was_active || matches!(self.stage, Stage::Recording(_)))
        {
            self.stage = Stage::Idle;
            self.hold = None;
        }
    }

    fn on_processing_finished(&mut self) -> Option<Effect> {
        self.stage = Stage::Idle;
        self.hold = None;
        let pending = self.pending_press.take()?;
        debug!(
            "Pipeline drained; starting remembered press for '{}'",
            pending.binding_id
        );
        Some(self.begin_recording(
            pending.binding_id,
            pending.hotkey_string,
            pending.pressed_at,
            pending.locked,
        ))
    }

    /// Reconcile the optimistic `Stage::Recording` after the executor reports
    /// whether recording actually began (microphone access can be denied).
    fn on_start_result(&mut self, binding_id: &str, started: bool) {
        if !started && matches!(&self.stage, Stage::Recording(id) if id == binding_id) {
            self.stage = Stage::Idle;
            self.hold = None;
        }
    }

    /// Optimistic transition to `Recording`; rolled back via
    /// [`CoordinatorState::on_start_result`] if the effect fails to start
    /// recording for real.
    fn begin_recording(
        &mut self,
        binding_id: String,
        hotkey_string: String,
        pressed_at: Instant,
        locked: bool,
    ) -> Effect {
        self.stage = Stage::Recording(binding_id.clone());
        self.hold = Some(Hold { pressed_at, locked });
        Effect::Start {
            binding_id,
            hotkey_string,
        }
    }

    fn begin_processing(&mut self, binding_id: String, hotkey_string: String) -> Effect {
        self.stage = Stage::Processing;
        self.hold = None;
        Effect::Stop {
            binding_id,
            hotkey_string,
        }
    }
}

/// Serialises all transcription lifecycle events through a single thread
/// to eliminate race conditions between keyboard shortcuts, signals, and
/// the async transcribe-paste pipeline. The thread is a thin shell: it
/// transports commands to the pure [`CoordinatorState`] and executes the
/// returned [`Effect`]s.
pub struct TranscriptionCoordinator {
    tx: Sender<Command>,
}

pub fn is_transcribe_binding(id: &str) -> bool {
    id == "transcribe" || id == "transcribe_with_post_process"
}

impl TranscriptionCoordinator {
    pub fn new(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut state = CoordinatorState::new();

                loop {
                    let cmd = if let Some(deadline) = state.grace_deadline() {
                        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                            Ok(cmd) => cmd,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if let Some(effect) = state.on_grace_expired() {
                                    run_effect(&app, &mut state, effect);
                                }
                                continue;
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    } else {
                        match rx.recv() {
                            Ok(cmd) => cmd,
                            Err(_) => break,
                        }
                    };

                    match cmd {
                        Command::Input(input) => {
                            if let Some(effect) = state.on_input(input, Instant::now()) {
                                run_effect(&app, &mut state, effect);
                            }
                        }
                        Command::Cancel {
                            recording_was_active,
                        } => state.on_cancel(recording_was_active),
                        Command::ProcessingFinished => {
                            if let Some(effect) = state.on_processing_finished() {
                                run_effect(&app, &mut state, effect);
                            }
                        }
                    }
                }
                debug!("Transcription coordinator exited");
            }));
            if let Err(e) = result {
                error!("Transcription coordinator panicked: {e:?}");
            }
        });

        Self { tx }
    }

    /// Send a keyboard input event for a transcribe binding. `hold_threshold`
    /// only matters for [`ShortcutActivation::HoldOrToggle`].
    pub fn send_input(
        &self,
        binding_id: &str,
        hotkey_string: &str,
        is_pressed: bool,
        mode: ShortcutActivation,
        hold_threshold: Duration,
    ) {
        self.send(
            binding_id,
            hotkey_string,
            is_pressed,
            mode,
            hold_threshold,
            false,
        );
    }

    /// Send an external trigger (SIGUSR2, CLI flag). Always a toggle press,
    /// always exempt from debounce — see [`InputEvent::external`].
    pub fn send_external_input(&self, binding_id: &str, source: &str) {
        self.send(
            binding_id,
            source,
            true,
            ShortcutActivation::Toggle,
            Duration::ZERO,
            true,
        );
    }

    fn send(
        &self,
        binding_id: &str,
        hotkey_string: &str,
        is_pressed: bool,
        mode: ShortcutActivation,
        hold_threshold: Duration,
        external: bool,
    ) {
        if self
            .tx
            .send(Command::Input(InputEvent {
                binding_id: binding_id.to_string(),
                hotkey_string: hotkey_string.to_string(),
                is_pressed,
                mode,
                hold_threshold,
                external,
            }))
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_cancel(&self, recording_was_active: bool) {
        if self
            .tx
            .send(Command::Cancel {
                recording_was_active,
            })
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_processing_finished(&self) {
        if self.tx.send(Command::ProcessingFinished).is_err() {
            warn!("Transcription coordinator channel closed");
        }
    }
}

fn run_effect(app: &AppHandle, state: &mut CoordinatorState, effect: Effect) {
    match effect {
        Effect::Start {
            binding_id,
            hotkey_string,
        } => {
            let started = start(app, &binding_id, &hotkey_string);
            state.on_start_result(&binding_id, started);
        }
        Effect::Stop {
            binding_id,
            hotkey_string,
        } => stop(app, &binding_id, &hotkey_string),
    }
}

/// Execute a start effect; returns whether recording actually began, so the
/// state machine can roll back its optimistic transition on failure.
fn start(app: &AppHandle, binding_id: &str, hotkey_string: &str) -> bool {
    let Some(action) = ACTION_MAP.get(binding_id) else {
        warn!("No action in ACTION_MAP for '{binding_id}'");
        return false;
    };
    action.start(app, binding_id, hotkey_string);
    let recording = app
        .try_state::<Arc<AudioRecordingManager>>()
        .is_some_and(|a| a.is_recording());
    if !recording {
        debug!("Start for '{binding_id}' did not begin recording; staying idle");
    }
    recording
}

fn stop(app: &AppHandle, binding_id: &str, hotkey_string: &str) {
    let Some(action) = ACTION_MAP.get(binding_id) else {
        warn!("No action in ACTION_MAP for '{binding_id}'");
        return;
    };
    action.stop(app, binding_id, hotkey_string);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_target(name: &str) -> crate::paste_tx::TargetIdentity {
        crate::paste_tx::TargetIdentity::for_test(name)
    }

    fn live_ledger() -> (LiveInsertionLedger, crate::paste_tx::TargetIdentity) {
        let target = live_target("target-a");
        (
            LiveInsertionLedger::new(
                InsertionMode::LiveCommittedExperimental,
                Some(target.clone()),
            ),
            target,
        )
    }

    fn confirm_inserted(ledger: &mut LiveInsertionLedger, attempt: LiveInsertionAttempt) {
        assert!(ledger.record_attempt_result(attempt.sequence, LiveInsertionOutcome::Inserted));
    }

    #[test]
    fn insertion_mode_defaults_to_at_stop_and_non_live_modes_never_plan_deltas() {
        assert_eq!(InsertionMode::default(), InsertionMode::AtStop);
        let target = live_target("target-a");
        for mode in [InsertionMode::AtStop, InsertionMode::PreviewOnly] {
            let mut ledger = LiveInsertionLedger::new(mode, Some(target.clone()));
            assert!(ledger.observe_speech_evidence(Some(&target)).is_none());
            assert!(ledger.observe_committed("hello", Some(&target)).is_none());
            assert!(ledger.finalize("hello", Some(&target)).is_none());
            assert!(ledger.attempts().is_empty());
            assert_eq!(ledger.mode(), mode);
        }
    }

    #[test]
    fn repeated_committed_events_insert_only_new_delta_without_duplicates() {
        let (mut ledger, target) = live_ledger();
        assert!(ledger.observe_speech_evidence(Some(&target)).is_none());

        let first = ledger
            .observe_committed("hello", Some(&target))
            .expect("first committed delta");
        assert_eq!(first.text, "hello");
        confirm_inserted(&mut ledger, first);

        assert!(ledger.observe_committed("hello", Some(&target)).is_none());
        let second = ledger
            .observe_committed("hello world", Some(&target))
            .expect("only the new suffix");
        assert_eq!(second.text, " world");
        confirm_inserted(&mut ledger, second);

        assert_eq!(ledger.inserted_committed(), "hello world");
        assert_eq!(ledger.attempts().len(), 2);
        assert_eq!(ledger.attempts()[0].byte_len, 5);
        assert_eq!(ledger.attempts()[1].byte_len, 6);
    }

    #[test]
    fn revised_already_inserted_prefix_stops_live_insertion_without_replacement() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let first = ledger
            .observe_committed("hello world", Some(&target))
            .expect("initial committed text");
        confirm_inserted(&mut ledger, first);

        assert!(ledger
            .observe_committed("hello there", Some(&target))
            .is_none());
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::RevisedInsertedPrefix)
        );
        assert!(ledger.finalize("hello there!", Some(&target)).is_none());
        assert_eq!(ledger.inserted_committed(), "hello world");
        assert_eq!(ledger.attempts().len(), 1);
    }

    #[test]
    fn foreground_target_change_permanently_stops_session_insertion() {
        let (mut ledger, target) = live_ledger();
        let other = live_target("target-b");
        ledger.observe_speech_evidence(Some(&target));

        assert!(ledger.observe_committed("hello", Some(&other)).is_none());
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::TargetChanged)
        );
        assert_eq!(
            ledger.safety_events(),
            &[LiveInsertionStopReason::TargetChanged]
        );
        assert!(ledger
            .observe_committed("hello world", Some(&target))
            .is_none());
        assert!(ledger.attempts().is_empty());
    }

    #[test]
    fn missing_target_identity_fails_closed_before_first_insertion() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        assert!(ledger.observe_committed("hello", None).is_none());
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::TargetLost)
        );
        assert!(ledger.attempts().is_empty());
    }

    #[test]
    fn committed_hallucination_waits_for_positive_speech_evidence_latch() {
        let (mut ledger, target) = live_ledger();
        assert!(ledger
            .observe_committed("thank you for watching", Some(&target))
            .is_none());
        assert!(ledger.attempts().is_empty());
        assert_eq!(ledger.pending_text(), "thank you for watching");

        let first = ledger
            .observe_speech_evidence(Some(&target))
            .expect("speech latch makes existing committed text eligible");
        assert_eq!(first.text, "thank you for watching");
        confirm_inserted(&mut ledger, first);
    }

    #[test]
    fn cancellation_never_schedules_or_flushes_a_final_tail() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let first = ledger
            .observe_committed("hello", Some(&target))
            .expect("initial delta");
        confirm_inserted(&mut ledger, first);

        ledger.cancel();
        assert!(ledger.is_cancelled());
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::Cancelled)
        );
        assert!(ledger
            .observe_committed("hello world", Some(&target))
            .is_none());
        assert_eq!(ledger.pending_text(), "");
        assert!(ledger.finalize("hello world", Some(&target)).is_none());
        assert!(!ledger.final_tail_flushed());
        assert_eq!(ledger.inserted_committed(), "hello");
    }

    #[test]
    fn target_change_after_planning_before_injection_is_terminal() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let attempt = ledger
            .observe_committed("hello", Some(&target))
            .expect("initial delta");
        assert_eq!(attempt.target, target);

        assert!(ledger.record_attempt_result(attempt.sequence, LiveInsertionOutcome::TargetChanged));
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::TargetChanged)
        );
        assert_eq!(ledger.inserted_committed(), "");
        assert!(ledger
            .observe_committed("hello world", Some(&target))
            .is_none());
        assert_eq!(ledger.attempts().len(), 1);
    }

    #[test]
    fn finalization_flushes_remaining_tail_exactly_once() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let first = ledger
            .observe_committed("hello", Some(&target))
            .expect("initial delta");
        confirm_inserted(&mut ledger, first);

        let tail = ledger
            .finalize("hello world", Some(&target))
            .expect("remaining final tail");
        assert_eq!(tail.kind, LiveInsertionAttemptKind::FinalTail);
        assert_eq!(tail.text, " world");
        assert!(ledger.finalize("hello world", Some(&target)).is_none());
        confirm_inserted(&mut ledger, tail);
        assert!(ledger.final_tail_flushed());
        assert_eq!(ledger.inserted_committed(), "hello world");
        assert!(ledger.finalize("hello world", Some(&target)).is_none());
    }

    #[test]
    fn input_failure_is_terminal_and_never_retries_uncertain_text() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let attempt = ledger
            .observe_committed("hello", Some(&target))
            .expect("initial delta");
        assert!(ledger.record_attempt_result(attempt.sequence, LiveInsertionOutcome::InputFailed));
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::InputFailed)
        );
        assert!(ledger
            .observe_committed("hello world", Some(&target))
            .is_none());
        assert_eq!(ledger.attempts().len(), 1);
    }

    #[test]
    fn clipboard_ownership_loss_stops_live_insertion_without_advancing_prefix() {
        let (mut ledger, target) = live_ledger();
        ledger.observe_speech_evidence(Some(&target));
        let attempt = ledger
            .observe_committed("hello", Some(&target))
            .expect("initial delta");
        assert!(ledger.record_attempt_result(
            attempt.sequence,
            LiveInsertionOutcome::ClipboardOwnershipLost
        ));
        assert_eq!(
            ledger.stop_reason(),
            Some(LiveInsertionStopReason::ClipboardOwnershipLost)
        );
        assert_eq!(ledger.inserted_committed(), "");
        assert!(ledger
            .observe_committed("hello world", Some(&target))
            .is_none());
    }

    #[test]
    fn push_to_talk_release_while_recording_defers_release() {
        assert_eq!(
            classify_ptt_event(None, false, true, "transcribe", Some("transcribe")),
            PttAction::DeferRelease
        );
    }

    #[test]
    fn push_to_talk_press_matching_pending_release_cancels_release() {
        assert_eq!(
            classify_ptt_event(
                Some("transcribe"),
                true,
                true,
                "transcribe",
                Some("transcribe")
            ),
            PttAction::CancelRelease
        );
    }

    #[test]
    fn toggle_mode_press_and_release_pass_through() {
        assert_eq!(
            classify_ptt_event(
                Some("transcribe"),
                true,
                false,
                "transcribe",
                Some("transcribe")
            ),
            PttAction::Passthrough
        );
        assert_eq!(
            classify_ptt_event(None, false, false, "transcribe", Some("transcribe")),
            PttAction::Passthrough
        );
    }

    #[test]
    fn press_for_different_binding_than_pending_release_passes_through() {
        assert_eq!(
            classify_ptt_event(
                Some("transcribe"),
                true,
                true,
                "transcribe_with_post_process",
                Some("transcribe")
            ),
            PttAction::Passthrough
        );
    }

    #[test]
    fn press_matching_pending_release_cancels_without_recording_state() {
        assert_eq!(
            classify_ptt_event(Some("transcribe"), true, true, "transcribe", None),
            PttAction::CancelRelease
        );
    }

    // ---------------------------------------------------------------------
    // Busy-pipeline input classification.
    //
    // Toggle-style triggers (SIGUSR2, CLI flags, pedals that signal on both
    // edges) flip state on every edge. Dropping a press that arrives while
    // the previous pipeline is still processing desyncs the parity: the next
    // edge then starts a recording no one will stop, leaving the overlay
    // waiting for input with the button long released.
    // ---------------------------------------------------------------------

    #[test]
    fn toggle_press_during_processing_remembers_start() {
        assert_eq!(
            classify_busy_input(true, ShortcutActivation::Toggle, None),
            BusyAction::Remember
        );
    }

    #[test]
    fn second_toggle_press_during_processing_forgets_press() {
        assert_eq!(
            classify_busy_input(true, ShortcutActivation::Toggle, Some(Remembered::Locked)),
            BusyAction::Forget
        );
    }

    #[test]
    fn toggle_release_during_processing_is_ignored() {
        assert_eq!(
            classify_busy_input(false, ShortcutActivation::Toggle, None),
            BusyAction::Ignore
        );
        assert_eq!(
            classify_busy_input(false, ShortcutActivation::Toggle, Some(Remembered::Locked)),
            BusyAction::Ignore
        );
    }

    #[test]
    fn hold_modes_classify_busy_inputs_by_pending_state() {
        let cases = [
            (true, None, BusyAction::Remember),
            (true, Some(Remembered::Held), BusyAction::Ignore),
            (true, Some(Remembered::Locked), BusyAction::Forget),
            (false, None, BusyAction::Ignore),
            (false, Some(Remembered::Held), BusyAction::Ignore),
            (false, Some(Remembered::Locked), BusyAction::Ignore),
        ];

        for mode in [
            ShortcutActivation::PushToTalk,
            ShortcutActivation::HoldOrToggle,
        ] {
            for (is_pressed, remembered, expected) in cases {
                assert_eq!(classify_busy_input(is_pressed, mode, remembered), expected);
            }
        }
    }

    /// Toggle parity across a busy window: an odd number of presses remembers
    /// one start, each further press flips the remembered press off/on again.
    #[test]
    fn toggle_presses_alternate_remember_and_forget_while_busy() {
        let mut remembered = None;
        for expected in [
            BusyAction::Remember,
            BusyAction::Forget,
            BusyAction::Remember,
        ] {
            let action = classify_busy_input(true, ShortcutActivation::Toggle, remembered);
            assert_eq!(action, expected);
            remembered = (action == BusyAction::Remember).then_some(Remembered::Locked);
        }
        assert!(remembered.is_some());
    }

    // ---------------------------------------------------------------------
    // Sequence-level regression coverage for issue #1539.
    //
    // Under X11 key auto-repeat, holding a push-to-talk key does not emit one
    // long press. It emits the initial press followed by a stream of
    // synthesized release/press pairs, then a single genuine release on key-up.
    // Before the fix, every synthesized release passed straight through and
    // stopped recording, so holding the key "rapidly toggled" recording on and
    // off. The fix defers each release for a short grace window and cancels it
    // when the matching auto-repeat press arrives.
    //
    // The unit tests above assert the classifiers in isolation. The harness
    // below drives the real `CoordinatorState` through whole event sequences
    // — the same `on_input` / `on_grace_expired` handlers the coordinator
    // thread runs — so a burst can be exercised deterministically without a
    // Tauri AppHandle or real timers, and the tests can never drift from the
    // production transitions.
    // ---------------------------------------------------------------------

    const BINDING: &str = "transcribe";

    #[derive(Clone, Copy)]
    enum Ev {
        /// A key-down event (real initial press or a synthesized auto-repeat press).
        Press,
        /// A key-up event (synthesized auto-repeat release or the genuine key-up).
        Release,
        /// The `RELEASE_GRACE` window elapsed with no cancelling press arriving.
        Grace,
    }

    struct DriveResult {
        starts: u32,
        stops: u32,
        stage: Stage,
    }

    fn ptt_input(is_pressed: bool) -> InputEvent {
        InputEvent {
            binding_id: BINDING.to_string(),
            hotkey_string: BINDING.to_string(),
            is_pressed,
            mode: ShortcutActivation::PushToTalk,
            hold_threshold: Duration::ZERO,
            external: false,
        }
    }

    /// Feeds an event sequence to a real [`CoordinatorState`] the way the
    /// coordinator thread would; effects are counted instead of executed.
    fn drive(events: &[Ev]) -> DriveResult {
        let mut state = CoordinatorState::new();
        let mut clock = Instant::now();
        let mut starts = 0u32;
        let mut stops = 0u32;

        for ev in events {
            // Auto-repeat events arrive a few ms apart, well inside DEBOUNCE.
            clock += Duration::from_millis(5);

            let effect = match ev {
                Ev::Grace => state.on_grace_expired(),
                Ev::Press | Ev::Release => {
                    state.on_input(ptt_input(matches!(ev, Ev::Press)), clock)
                }
            };
            match effect {
                Some(Effect::Start { .. }) => starts += 1,
                Some(Effect::Stop { .. }) => stops += 1,
                None => {}
            }
        }

        DriveResult {
            starts,
            stops,
            stage: state.stage,
        }
    }

    /// Initial press plus several synthesized release/press pairs, as X11 emits
    /// while a push-to-talk key is held down.
    fn autorepeat_burst() -> Vec<Ev> {
        let mut events = vec![Ev::Press];
        for _ in 0..6 {
            events.push(Ev::Release);
            events.push(Ev::Press);
        }
        events
    }

    /// Regression for #1539: a burst of X11 auto-repeat release/press pairs must
    /// not stop recording. Before the fix the first synthesized release stopped
    /// recording immediately (stops == 1, stage left Recording), which produced
    /// the rapid on/off toggling. With the fix the releases are coalesced and
    /// recording stays continuously active for the whole burst.
    #[test]
    fn x11_autorepeat_burst_does_not_toggle_recording() {
        let result = drive(&autorepeat_burst());
        assert_eq!(result.starts, 1, "recording should start exactly once");
        assert_eq!(
            result.stops, 0,
            "synthesized auto-repeat releases must not stop recording mid-burst"
        );
        assert_eq!(
            result.stage,
            Stage::Recording(BINDING.to_string()),
            "recording must remain active across the entire auto-repeat burst"
        );
    }

    /// Complements the burst test: once the key is genuinely released and the
    /// grace window elapses with no re-press, recording stops exactly once. This
    /// proves the debounce only coalesces synthesized releases and does not wedge
    /// the coordinator or swallow the real key-up.
    #[test]
    fn genuine_release_after_grace_stops_recording_once() {
        let mut events = autorepeat_burst();
        events.push(Ev::Release); // genuine key-up
        events.push(Ev::Grace); // grace window elapses, no cancelling press
        let result = drive(&events);
        assert_eq!(result.starts, 1, "recording should start exactly once");
        assert_eq!(
            result.stops, 1,
            "a genuine release should stop recording exactly once"
        );
        assert_eq!(result.stage, Stage::Processing);
    }

    // ---------------------------------------------------------------------
    // Sequence-level coverage of the busy-pipeline and cancel paths, driven
    // through the real machine.
    // ---------------------------------------------------------------------

    /// PTT press while the pipeline is busy is remembered and starts recording
    /// once the pipeline drains.
    #[test]
    fn press_during_processing_starts_after_drain() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();

        let effect = state.on_input(ptt_input(true), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));

        let effect = state.on_input(ptt_input(false), now + Duration::from_millis(100));
        assert!(effect.is_none(), "release should be deferred, not fired");

        let effect = state.on_grace_expired();
        assert!(matches!(effect, Some(Effect::Stop { .. })));

        let effect = state.on_input(ptt_input(true), now + Duration::from_millis(200));
        assert!(effect.is_none(), "busy pipeline must remember, not start");

        let effect = state.on_processing_finished();
        assert!(
            matches!(effect, Some(Effect::Start { .. })),
            "remembered press should start once the pipeline drains"
        );
    }

    /// Two toggle presses inside one busy window net to no-op: nothing starts
    /// when the pipeline drains (toggle parity).
    #[test]
    fn toggle_presses_during_processing_net_noop_after_drain() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();

        let effect = state.on_input(ptt_input(true), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));
        let effect = state.on_input(ptt_input(false), now + Duration::from_millis(100));
        assert!(effect.is_none());
        let effect = state.on_grace_expired();
        assert!(matches!(effect, Some(Effect::Stop { .. })));

        let toggle = |state: &mut CoordinatorState, at: Instant| {
            state.on_input(
                InputEvent {
                    binding_id: BINDING.to_string(),
                    hotkey_string: BINDING.to_string(),
                    is_pressed: true,
                    mode: ShortcutActivation::Toggle,
                    hold_threshold: Duration::ZERO,
                    external: true,
                },
                at,
            )
        };

        let effect = toggle(&mut state, now + Duration::from_millis(200));
        assert!(effect.is_none());
        let effect = toggle(&mut state, now + Duration::from_millis(300));
        assert!(effect.is_none());

        let effect = state.on_processing_finished();
        assert!(
            effect.is_none(),
            "even number of busy toggle presses must not start recording"
        );
        assert_eq!(state.stage, Stage::Idle);
    }

    /// Cancel while processing abandons a remembered press: the pipeline drains
    /// to idle and nothing starts.
    #[test]
    fn cancel_during_processing_drops_remembered_press() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();

        let effect = state.on_input(ptt_input(true), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));
        let effect = state.on_input(ptt_input(false), now + Duration::from_millis(100));
        assert!(effect.is_none());
        let effect = state.on_grace_expired();
        assert!(matches!(effect, Some(Effect::Stop { .. })));

        let effect = state.on_input(ptt_input(true), now + Duration::from_millis(200));
        assert!(effect.is_none());

        state.on_cancel(false);
        assert_eq!(
            state.stage,
            Stage::Processing,
            "cancel must not reset mid-processing — the pipeline still finishes"
        );

        let effect = state.on_processing_finished();
        assert!(
            effect.is_none(),
            "cancelled session must not spawn a deferred recording"
        );
        assert_eq!(state.stage, Stage::Idle);
    }

    fn toggle_input(external: bool) -> InputEvent {
        toggle_input_for(BINDING, external)
    }

    fn toggle_input_for(binding_id: &str, external: bool) -> InputEvent {
        InputEvent {
            binding_id: binding_id.to_string(),
            hotkey_string: binding_id.to_string(),
            is_pressed: true,
            mode: ShortcutActivation::Toggle,
            hold_threshold: Duration::ZERO,
            external,
        }
    }

    /// Start and stop one toggle recording so the machine sits in `Processing`.
    fn drive_into_processing(state: &mut CoordinatorState, now: Instant) {
        let effect = state.on_input(toggle_input(true), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));
        let effect = state.on_input(toggle_input(true), now + Duration::from_millis(100));
        assert!(matches!(effect, Some(Effect::Stop { .. })));
        assert_eq!(state.stage, Stage::Processing);
    }

    const OTHER_BINDING: &str = "transcribe_with_post_process";

    /// Only one press can be pending. Once a binding has claimed it, a toggle
    /// for a different binding is ignored (as it is while recording) instead of
    /// replacing the remembered press, so the pending binding's parity holds:
    /// two transcribe toggles still net to no-op.
    #[test]
    fn different_binding_does_not_replace_pending_press() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();
        drive_into_processing(&mut state, now);

        let at = |ms| now + Duration::from_millis(ms);
        assert!(state.on_input(toggle_input(true), at(200)).is_none());
        assert!(state
            .on_input(toggle_input_for(OTHER_BINDING, true), at(300))
            .is_none());
        assert!(state.on_input(toggle_input(true), at(400)).is_none());

        let effect = state.on_processing_finished();
        assert!(
            effect.is_none(),
            "two transcribe toggles net to no-op; the ignored post-process toggle must not start"
        );
        assert_eq!(state.stage, Stage::Idle);
    }

    /// The binding that claimed the pending press is the one that starts on
    /// drain, regardless of other bindings toggled in between.
    #[test]
    fn drain_starts_the_pending_binding_not_a_later_one() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();
        drive_into_processing(&mut state, now);

        let at = |ms| now + Duration::from_millis(ms);
        assert!(state.on_input(toggle_input(true), at(200)).is_none());
        assert!(state
            .on_input(toggle_input_for(OTHER_BINDING, true), at(300))
            .is_none());

        match state.on_processing_finished() {
            Some(Effect::Start { binding_id, .. }) => assert_eq!(binding_id, BINDING),
            other => panic!("expected Start for '{BINDING}', got {other:?}"),
        }
    }

    /// External triggers fire on every edge by design (e.g. SIGUSR2 sent on
    /// both key press and release). Two edges inside the debounce window must
    /// both be honoured, or the parity desyncs and recording wedges on.
    #[test]
    fn external_edges_inside_debounce_window_are_not_dropped() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();

        let effect = state.on_input(toggle_input(true), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));

        let effect = state.on_input(toggle_input(true), now + Duration::from_millis(5));
        assert!(
            matches!(effect, Some(Effect::Stop { .. })),
            "second external edge inside DEBOUNCE must stop the recording"
        );
        assert_eq!(state.stage, Stage::Processing);
    }

    /// Physical keyboard presses keep the debounce: a repeat inside the window
    /// is still dropped and recording stays active.
    #[test]
    fn keyboard_press_inside_debounce_window_is_still_dropped() {
        let mut state = CoordinatorState::new();
        let now = Instant::now();

        let effect = state.on_input(toggle_input(false), now);
        assert!(matches!(effect, Some(Effect::Start { .. })));

        let effect = state.on_input(toggle_input(false), now + Duration::from_millis(5));
        assert!(
            effect.is_none(),
            "keyboard repeat inside DEBOUNCE must be debounced"
        );
        assert_eq!(state.stage, Stage::Recording(BINDING.to_string()));
    }

    /// If the start effect fails to begin recording (e.g. microphone access
    /// denied), the optimistic transition rolls back to idle.
    #[test]
    fn failed_start_rolls_back_to_idle() {
        let mut state = CoordinatorState::new();

        let effect = state.on_input(ptt_input(true), Instant::now());
        assert!(matches!(effect, Some(Effect::Start { .. })));

        state.on_start_result(BINDING, false);
        assert_eq!(state.stage, Stage::Idle);
    }

    // ---------------------------------------------------------------------
    // Hold-or-toggle (the combined mode from #147) and the two legacy modes,
    // driven through the real machine on a synthetic clock. Recording starts
    // on key-down in every mode; the tests pin how each mode ends it.
    // ---------------------------------------------------------------------

    const HOLD_THRESHOLD: Duration = Duration::from_millis(300);

    fn input(mode: ShortcutActivation, is_pressed: bool) -> InputEvent {
        InputEvent {
            binding_id: BINDING.to_string(),
            hotkey_string: BINDING.to_string(),
            is_pressed,
            mode,
            hold_threshold: HOLD_THRESHOLD,
            external: false,
        }
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Hold-or-toggle: a key held past the threshold is push-to-talk — the
    /// (deferred) release stops recording.
    #[test]
    fn hold_or_toggle_long_hold_stops_on_release() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        assert!(matches!(
            state.on_input(input(mode, true), t0),
            Some(Effect::Start { .. })
        ));
        assert!(state.on_input(input(mode, false), t0 + ms(800)).is_none());
        assert!(
            matches!(state.on_grace_expired(), Some(Effect::Stop { .. })),
            "an 800ms hold must stop when its release grace elapses"
        );
        assert_eq!(state.stage, Stage::Processing);
    }

    /// Hold-or-toggle: a tap keeps recording (locked on); the next press stops.
    #[test]
    fn hold_or_toggle_tap_locks_recording_until_next_press() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        assert!(matches!(
            state.on_input(input(mode, true), t0),
            Some(Effect::Start { .. })
        ));
        assert!(state.on_input(input(mode, false), t0 + ms(120)).is_none());
        assert!(
            state.on_grace_expired().is_none(),
            "a 120ms tap must not stop recording"
        );
        assert_eq!(state.stage, Stage::Recording(BINDING.to_string()));
        assert!(state.is_locked());

        // Seconds later the user presses again to finish.
        assert!(matches!(
            state.on_input(input(mode, true), t0 + ms(5000)),
            Some(Effect::Stop { .. })
        ));
        assert_eq!(state.stage, Stage::Processing);
        // The release of that stopping press lands in the busy window and is
        // ignored, so nothing is remembered for the drain.
        assert!(state.on_input(input(mode, false), t0 + ms(5080)).is_none());
        assert!(state.on_processing_finished().is_none());
        assert_eq!(state.stage, Stage::Idle);
    }

    /// Hold-or-toggle: a locked session ignores stray releases — only a press
    /// ends it.
    #[test]
    fn hold_or_toggle_locked_session_ignores_release() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        state.on_input(input(mode, true), t0);
        state.on_input(input(mode, false), t0 + ms(100));
        assert!(state.on_grace_expired().is_none());
        assert!(state.is_locked());

        assert!(state.on_input(input(mode, false), t0 + ms(900)).is_none());
        assert!(
            state.grace_deadline().is_none(),
            "no release may be deferred once locked"
        );
        assert_eq!(state.stage, Stage::Recording(BINDING.to_string()));
    }

    /// Hold-or-toggle: while the key is genuinely held, extra presses do not
    /// stop the recording (that is the release's job).
    #[test]
    fn hold_or_toggle_press_while_held_is_ignored() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        state.on_input(input(mode, true), t0);
        assert!(state.on_input(input(mode, true), t0 + ms(400)).is_none());
        assert_eq!(state.stage, Stage::Recording(BINDING.to_string()));
        assert!(!state.is_locked());
    }

    /// Hold-or-toggle under X11 auto-repeat: the synthesized release/press
    /// pairs must not be misread as taps. The hold is measured from the
    /// original key-down to the genuine key-up.
    #[test]
    fn hold_or_toggle_autorepeat_burst_is_one_long_hold() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        let mut clock = t0;

        assert!(matches!(
            state.on_input(input(mode, true), clock),
            Some(Effect::Start { .. })
        ));
        // ~600ms of auto-repeat pairs a few ms apart.
        for _ in 0..60 {
            clock += ms(5);
            assert!(state.on_input(input(mode, false), clock).is_none());
            clock += ms(5);
            assert!(state.on_input(input(mode, true), clock).is_none());
            assert!(
                state.grace_deadline().is_none(),
                "auto-repeat press must cancel the deferred release"
            );
        }
        assert!(!state.is_locked(), "no tap may be classified mid-burst");

        clock += ms(5);
        assert!(state.on_input(input(mode, false), clock).is_none());
        assert!(
            matches!(state.on_grace_expired(), Some(Effect::Stop { .. })),
            "the genuine release after a ~600ms hold must stop recording"
        );
    }

    /// Hold-or-toggle: a press remembered during the busy window is measured
    /// from the real key-down, so a hold that straddles the drain still counts
    /// as a hold when it is released shortly after recording actually starts.
    #[test]
    fn hold_or_toggle_remembered_press_measures_hold_from_real_key_down() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        // Previous session: hold, release, stop → Processing.
        state.on_input(input(mode, true), t0);
        state.on_input(input(mode, false), t0 + ms(800));
        assert!(matches!(
            state.on_grace_expired(),
            Some(Effect::Stop { .. })
        ));

        // Pressed again while busy; still held when the pipeline drains 700ms later.
        assert!(state.on_input(input(mode, true), t0 + ms(1000)).is_none());
        assert!(matches!(
            state.on_processing_finished(),
            Some(Effect::Start { .. })
        ));
        // Released 100ms after recording began — but 800ms after key-down.
        assert!(state.on_input(input(mode, false), t0 + ms(1800)).is_none());
        assert!(
            matches!(state.on_grace_expired(), Some(Effect::Stop { .. })),
            "held 800ms overall: must stop, not lock"
        );
    }

    /// Toggle: releases never stop, the next press does. (Toggle is the
    /// combined machine with the session locked from the start.)
    #[test]
    fn toggle_mode_ignores_release_and_stops_on_next_press() {
        let mode = ShortcutActivation::Toggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        assert!(matches!(
            state.on_input(input(mode, true), t0),
            Some(Effect::Start { .. })
        ));
        assert!(state.is_locked());
        assert!(state.on_input(input(mode, false), t0 + ms(100)).is_none());
        assert!(
            state.grace_deadline().is_none(),
            "toggle never defers releases"
        );
        assert!(state.on_input(input(mode, false), t0 + ms(3000)).is_none());
        assert!(matches!(
            state.on_input(input(mode, true), t0 + ms(4000)),
            Some(Effect::Stop { .. })
        ));
    }

    /// Push-to-talk: even a very short press stops on release — there is no
    /// tap-to-lock in this mode (hold threshold of zero).
    #[test]
    fn push_to_talk_short_press_still_stops_on_release() {
        let mode = ShortcutActivation::PushToTalk;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        assert!(matches!(
            state.on_input(input(mode, true), t0),
            Some(Effect::Start { .. })
        ));
        assert!(state.on_input(input(mode, false), t0 + ms(40)).is_none());
        assert!(matches!(
            state.on_grace_expired(),
            Some(Effect::Stop { .. })
        ));
    }

    /// Cancel (Escape) during a locked hold-or-toggle session resets cleanly so
    /// the next press starts a fresh recording rather than stopping a dead one.
    #[test]
    fn hold_or_toggle_cancel_clears_locked_session() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        state.on_input(input(mode, true), t0);
        state.on_input(input(mode, false), t0 + ms(100));
        assert!(state.on_grace_expired().is_none());
        assert!(state.is_locked());

        state.on_cancel(true);
        assert_eq!(state.stage, Stage::Idle);
        assert!(!state.is_locked());
        assert!(matches!(
            state.on_input(input(mode, true), t0 + ms(2000)),
            Some(Effect::Start { .. })
        ));
    }

    /// Switching to toggle while an unlocked hold recording is running must not
    /// strand it: in toggle mode a press always stops.
    #[test]
    fn toggle_press_stops_recording_started_as_hold() {
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();

        state.on_input(input(ShortcutActivation::HoldOrToggle, true), t0);
        assert!(!state.is_locked());
        assert!(matches!(
            state.on_input(input(ShortcutActivation::Toggle, true), t0 + ms(2000)),
            Some(Effect::Stop { .. })
        ));
    }

    // Hold-vs-tap classification while the previous transcription is busy.
    fn hold_or_toggle_into_processing(state: &mut CoordinatorState, t0: Instant) {
        let mode = ShortcutActivation::HoldOrToggle;
        assert!(matches!(
            state.on_input(input(mode, true), t0),
            Some(Effect::Start { .. })
        ));
        assert!(state.on_input(input(mode, false), t0 + ms(800)).is_none());
        assert!(matches!(
            state.on_grace_expired(),
            Some(Effect::Stop { .. })
        ));
        assert_eq!(state.stage, Stage::Processing);
    }

    #[test]
    fn hold_or_toggle_tap_during_processing_queues_locked_start() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        assert!(state.on_input(input(mode, true), t0 + ms(1000)).is_none());
        assert!(state.on_input(input(mode, false), t0 + ms(1100)).is_none());
        assert!(state.on_grace_expired().is_none());
        assert!(state.is_locked(), "a busy tap should queue a locked start");
        assert!(matches!(
            state.on_processing_finished(),
            Some(Effect::Start { .. })
        ));
        assert!(state.is_locked());
    }

    #[test]
    fn hold_or_toggle_completed_hold_during_processing_nets_noop() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        assert!(state.on_input(input(mode, true), t0 + ms(1000)).is_none());
        assert!(state.on_input(input(mode, false), t0 + ms(1600)).is_none());
        assert!(state.on_grace_expired().is_none());
        assert!(!state.is_locked());

        assert!(
            state.on_processing_finished().is_none(),
            "a 600ms hold that ended before the drain has nothing left to start"
        );
        assert_eq!(state.stage, Stage::Idle);
    }

    #[test]
    fn hold_or_toggle_two_taps_during_processing_net_noop() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        assert!(state.on_input(input(mode, true), t0 + ms(1000)).is_none());
        assert!(state.on_input(input(mode, false), t0 + ms(1100)).is_none());
        assert!(state.on_grace_expired().is_none());
        assert!(state.is_locked());

        assert!(state.on_input(input(mode, true), t0 + ms(1500)).is_none());
        assert!(
            !state.is_locked(),
            "the second tap's press forgets the queued tap"
        );
        assert!(state.on_input(input(mode, false), t0 + ms(1600)).is_none());
        assert!(state.grace_deadline().is_none());

        assert!(state.on_processing_finished().is_none());
        assert_eq!(state.stage, Stage::Idle);
    }

    #[test]
    fn ptt_tap_inside_busy_window_nets_noop() {
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        assert!(state.on_input(ptt_input(true), t0 + ms(1000)).is_none());
        assert!(state.on_input(ptt_input(false), t0 + ms(1040)).is_none());
        assert!(state.on_grace_expired().is_none());
        assert!(state.on_processing_finished().is_none());
    }

    /// The pipeline drains inside the 50ms grace of a busy tap: recording
    /// starts first (unlocked, from the real key-down), and the grace then
    /// resolves against the live recording, locking it as the tap it was.
    #[test]
    fn hold_or_toggle_drain_inside_busy_release_grace_still_classifies_tap() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        assert!(state.on_input(input(mode, true), t0 + ms(1000)).is_none());
        assert!(state.on_input(input(mode, false), t0 + ms(1100)).is_none());
        assert!(matches!(
            state.on_processing_finished(),
            Some(Effect::Start { .. })
        ));
        assert!(!state.is_locked());

        assert!(state.on_grace_expired().is_none());
        assert!(state.is_locked(), "the deferred 100ms release is a tap");
    }

    /// X11 auto-repeat while busy, key still held at the drain: recording
    /// starts measured from the first press, not from the last synthesized
    /// press before the drain. Released 400ms after the real key-down but
    /// only ~100ms after the drain — a hold, so it must stop rather than lock.
    #[test]
    fn hold_or_toggle_autorepeat_burst_straddling_drain_measures_from_first_press() {
        let mode = ShortcutActivation::HoldOrToggle;
        let mut state = CoordinatorState::new();
        let t0 = Instant::now();
        hold_or_toggle_into_processing(&mut state, t0);

        let mut clock = t0 + ms(1000);
        assert!(state.on_input(input(mode, true), clock).is_none());
        for _ in 0..30 {
            clock += ms(5);
            assert!(state.on_input(input(mode, false), clock).is_none());
            clock += ms(5);
            assert!(state.on_input(input(mode, true), clock).is_none());
            assert!(state.grace_deadline().is_none());
        }

        // Drain at ~t0 + 1300ms with the key still down.
        assert!(matches!(
            state.on_processing_finished(),
            Some(Effect::Start { .. })
        ));
        assert!(!state.is_locked());

        for _ in 0..10 {
            clock += ms(5);
            assert!(state.on_input(input(mode, false), clock).is_none());
            clock += ms(5);
            assert!(state.on_input(input(mode, true), clock).is_none());
        }
        assert_eq!(clock, t0 + ms(1400));
        assert!(state.on_input(input(mode, false), clock).is_none());
        assert!(
            matches!(state.on_grace_expired(), Some(Effect::Stop { .. })),
            "held 400ms since the real key-down: must stop, not lock"
        );
        assert_eq!(state.stage, Stage::Processing);
    }
}
