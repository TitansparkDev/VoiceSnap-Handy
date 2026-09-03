use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};

const DEFAULT_MEDIA_CALL_TIMEOUT: Duration = Duration::from_millis(500);
const NO_ACTIVE_GENERATION: u64 = 0;

pub type MediaFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, MediaControlError>> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPlaybackState {
    Playing,
    Paused,
    Stopped,
    Unknown,
}

/// Snapshot of media state without retaining user-visible media metadata.
///
/// `session_key` and `state_revision` are intentionally opaque. A platform
/// backend may populate them when its native API exposes a stable current
/// session identity and a playback-state revision/timestamp. The controller
/// uses them only to avoid resuming a different or user-modified session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSnapshot {
    pub state: MediaPlaybackState,
    pub session_key: Option<u64>,
    pub state_revision: Option<u64>,
}

impl MediaSnapshot {
    pub const fn state_only(state: MediaPlaybackState) -> Self {
        Self {
            state,
            session_key: None,
            state_revision: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaControlError {
    Unavailable,
    Timeout,
    Failed(String),
}

impl std::fmt::Display for MediaControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => write!(f, "media control unavailable"),
            Self::Timeout => write!(f, "media control timed out"),
            Self::Failed(message) => write!(f, "media control failed: {message}"),
        }
    }
}

impl std::error::Error for MediaControlError {}

/// Platform adapter contract for the Wave 7 controller.
///
/// Implementations should use cancellable native async operations where
/// possible. The controller wraps every future in a timeout and drops it when
/// the deadline expires, so a backend must not perform a delayed side effect
/// after its future has been cancelled.
pub trait MediaBackend: Send + Sync + 'static {
    fn snapshot(&self) -> MediaFuture<'_, MediaSnapshot>;

    /// Pause current playback and return the post-action state.
    fn pause(&self) -> MediaFuture<'_, MediaSnapshot>;

    /// Resume current playback and return the post-action state.
    fn resume(&self) -> MediaFuture<'_, MediaSnapshot>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSession {
    generation: u64,
}

impl MediaSession {
    pub fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone)]
pub struct MediaSessionController {
    inner: Arc<ControllerInner>,
}

struct ControllerInner {
    next_generation: AtomicU64,
    active_generation: Arc<AtomicU64>,
    command_tx: mpsc::Sender<Command>,
}

enum Command {
    Begin(u64),
    End(u64),
    #[cfg(test)]
    Barrier(mpsc::Sender<()>),
}

#[derive(Default)]
struct SessionLedger {
    paused_owner: Option<u64>,
    pause_snapshot: Option<MediaSnapshot>,
}

impl MediaSessionController {
    pub fn new(backend: Arc<dyn MediaBackend>) -> Self {
        Self::with_timeout(backend, DEFAULT_MEDIA_CALL_TIMEOUT)
    }

    pub fn with_timeout(backend: Arc<dyn MediaBackend>, call_timeout: Duration) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let active_generation = Arc::new(AtomicU64::new(NO_ACTIVE_GENERATION));
        let worker_active_generation = active_generation.clone();

        if let Err(error) = thread::Builder::new()
            .name("media-session-controller".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        log::warn!("Media controller worker unavailable: {error}");
                        return;
                    }
                };

                let mut ledger = SessionLedger::default();
                while let Ok(command) = command_rx.recv() {
                    match command {
                        Command::Begin(generation) => runtime.block_on(handle_begin(
                            generation,
                            &worker_active_generation,
                            backend.as_ref(),
                            call_timeout,
                            &mut ledger,
                        )),
                        Command::End(generation) => runtime.block_on(handle_end(
                            generation,
                            &worker_active_generation,
                            backend.as_ref(),
                            call_timeout,
                            &mut ledger,
                        )),
                        #[cfg(test)]
                        Command::Barrier(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
        {
            // Media control is optional. If the worker cannot be created, the
            // channel simply becomes disconnected and recording continues.
            log::warn!("Failed to spawn media controller worker: {error}");
        }

        Self {
            inner: Arc::new(ControllerInner {
                next_generation: AtomicU64::new(1),
                active_generation,
                command_tx,
            }),
        }
    }

    /// Start a media-control generation without performing any platform work on
    /// the caller. This is safe to invoke from the recording/hotkey path.
    pub fn begin_session(&self) -> MediaSession {
        let generation = self.inner.next_generation.fetch_add(1, Ordering::AcqRel);
        self.inner
            .active_generation
            .store(generation, Ordering::Release);
        let _ = self.inner.command_tx.send(Command::Begin(generation));
        MediaSession { generation }
    }

    /// Finish a session. Only the currently active generation can make itself
    /// inactive; stale finishes are still queued so the worker can fence them
    /// against any transferred pause ownership.
    pub fn finish_session(&self, session: MediaSession) {
        let _ = self.inner.active_generation.compare_exchange(
            session.generation,
            NO_ACTIVE_GENERATION,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.inner.command_tx.send(Command::End(session.generation));
    }

    /// Cancellation follows the same ownership rule as a normal stop: resume
    /// only when this generation still owns a pause performed by the controller.
    pub fn cancel_session(&self, session: MediaSession) {
        self.finish_session(session);
    }

    #[cfg(test)]
    fn flush_for_test(&self) {
        let (done_tx, done_rx) = mpsc::channel();
        self.inner
            .command_tx
            .send(Command::Barrier(done_tx))
            .expect("media worker should still be running");
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("media worker did not process queued commands");
    }
}

async fn call_with_timeout<T>(
    timeout: Duration,
    future: MediaFuture<'_, T>,
) -> Result<T, MediaControlError> {
    match tokio::time::timeout(timeout, future).await {
        Ok(result) => result,
        Err(_) => Err(MediaControlError::Timeout),
    }
}

async fn handle_begin(
    generation: u64,
    active_generation: &AtomicU64,
    backend: &dyn MediaBackend,
    timeout: Duration,
    ledger: &mut SessionLedger,
) {
    // Process the generation fence even if this session has already ended by
    // the time its queued begin reaches the worker. Otherwise an older session
    // could retain pause ownership and later resume across a newer generation.
    if ledger.paused_owner.is_some() {
        ledger.paused_owner = Some(generation);
        log::debug!("Transferred media pause ownership to generation {generation}");
        return;
    }

    if active_generation.load(Ordering::Acquire) != generation {
        return;
    }

    let before = match call_with_timeout(timeout, backend.snapshot()).await {
        Ok(snapshot) => snapshot,
        Err(error) => {
            log_nonfatal("inspect before pause", generation, &error);
            return;
        }
    };

    if active_generation.load(Ordering::Acquire) != generation
        || before.state != MediaPlaybackState::Playing
    {
        return;
    }

    let paused = match call_with_timeout(timeout, backend.pause()).await {
        Ok(snapshot) if snapshot.state == MediaPlaybackState::Paused => snapshot,
        Ok(snapshot) => {
            log::debug!(
                "Media pause for generation {generation} returned state {:?}; leaving playback unowned",
                snapshot.state
            );
            return;
        }
        Err(error) => {
            log_nonfatal("pause", generation, &error);
            return;
        }
    };

    let active_now = active_generation.load(Ordering::Acquire);
    if active_now == NO_ACTIVE_GENERATION {
        // The recording ended while the async pause was in flight. We caused
        // the pause, so unwind it immediately, but only if state still matches
        // the post-pause observation.
        resume_if_unchanged(backend, timeout, generation, paused).await;
        return;
    }

    // An overlap may have become active while the pause future was running.
    // Transfer ownership directly to that generation so the old session can
    // never resume playback out from underneath the newer recording.
    ledger.paused_owner = Some(active_now);
    ledger.pause_snapshot = Some(paused);
    log::debug!("Paused media for recording generation {active_now}");
}

async fn handle_end(
    generation: u64,
    active_generation: &AtomicU64,
    backend: &dyn MediaBackend,
    timeout: Duration,
    ledger: &mut SessionLedger,
) {
    if ledger.paused_owner != Some(generation) {
        return;
    }

    let active_now = active_generation.load(Ordering::Acquire);
    if active_now != NO_ACTIVE_GENERATION && active_now != generation {
        // A newer generation is active even if its Begin command has not reached
        // the worker yet. Transfer ownership now so this stale end cannot resume.
        ledger.paused_owner = Some(active_now);
        return;
    }

    // If this generation somehow remains active, an end command raced ahead of
    // its caller-side generation fence. Do nothing rather than resume early.
    if active_now == generation {
        return;
    }

    let paused = ledger.pause_snapshot.take();
    ledger.paused_owner = None;

    if let Some(paused) = paused {
        resume_if_unchanged(backend, timeout, generation, paused).await;
    }
}

async fn resume_if_unchanged(
    backend: &dyn MediaBackend,
    timeout: Duration,
    generation: u64,
    paused: MediaSnapshot,
) {
    let current = match call_with_timeout(timeout, backend.snapshot()).await {
        Ok(snapshot) => snapshot,
        Err(error) => {
            log_nonfatal("inspect before resume", generation, &error);
            return;
        }
    };

    if current.state != MediaPlaybackState::Paused || media_state_was_changed(paused, current) {
        log::debug!(
            "Skipping media resume for generation {generation}: playback no longer matches the controller-owned pause"
        );
        return;
    }

    if let Err(error) = call_with_timeout(timeout, backend.resume()).await {
        log_nonfatal("resume", generation, &error);
    } else {
        log::debug!("Resumed media for recording generation {generation}");
    }
}

fn media_state_was_changed(paused: MediaSnapshot, current: MediaSnapshot) -> bool {
    if let (Some(paused_key), Some(current_key)) = (paused.session_key, current.session_key) {
        if paused_key != current_key {
            return true;
        }
    }

    if let (Some(paused_revision), Some(current_revision)) =
        (paused.state_revision, current.state_revision)
    {
        if paused_revision != current_revision {
            return true;
        }
    }

    false
}

fn log_nonfatal(action: &str, generation: u64, error: &MediaControlError) {
    log::debug!(
        "Media control {action} failed for recording generation {generation}; transcription continues: {error}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Mutex, MutexGuard},
        time::Instant,
    };

    #[derive(Clone)]
    struct MockBackend {
        inner: Arc<Mutex<MockState>>,
        delay: Duration,
    }

    struct MockState {
        snapshot: MediaSnapshot,
        unavailable: bool,
        pause_calls: usize,
        resume_calls: usize,
    }

    impl MockBackend {
        fn new(state: MediaPlaybackState) -> Self {
            Self {
                inner: Arc::new(Mutex::new(MockState {
                    snapshot: MediaSnapshot {
                        state,
                        session_key: Some(7),
                        state_revision: Some(1),
                    },
                    unavailable: false,
                    pause_calls: 0,
                    resume_calls: 0,
                })),
                delay: Duration::ZERO,
            }
        }

        fn unavailable() -> Self {
            let backend = Self::new(MediaPlaybackState::Unknown);
            backend.lock().unavailable = true;
            backend
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn lock(&self) -> MutexGuard<'_, MockState> {
            self.inner.lock().expect("mock media state poisoned")
        }

        fn manual_state_change(&self, state: MediaPlaybackState) {
            let mut inner = self.lock();
            inner.snapshot.state = state;
            inner.snapshot.state_revision = Some(inner.snapshot.state_revision.unwrap_or(0) + 1);
        }

        fn counts(&self) -> (usize, usize) {
            let inner = self.lock();
            (inner.pause_calls, inner.resume_calls)
        }

        async fn maybe_delay(&self) {
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
        }
    }

    impl MediaBackend for MockBackend {
        fn snapshot(&self) -> MediaFuture<'_, MediaSnapshot> {
            Box::pin(async move {
                self.maybe_delay().await;
                let inner = self.lock();
                if inner.unavailable {
                    Err(MediaControlError::Unavailable)
                } else {
                    Ok(inner.snapshot)
                }
            })
        }

        fn pause(&self) -> MediaFuture<'_, MediaSnapshot> {
            Box::pin(async move {
                self.maybe_delay().await;
                let mut inner = self.lock();
                if inner.unavailable {
                    return Err(MediaControlError::Unavailable);
                }
                inner.pause_calls += 1;
                inner.snapshot.state = MediaPlaybackState::Paused;
                inner.snapshot.state_revision =
                    Some(inner.snapshot.state_revision.unwrap_or(0) + 1);
                Ok(inner.snapshot)
            })
        }

        fn resume(&self) -> MediaFuture<'_, MediaSnapshot> {
            Box::pin(async move {
                self.maybe_delay().await;
                let mut inner = self.lock();
                if inner.unavailable {
                    return Err(MediaControlError::Unavailable);
                }
                inner.resume_calls += 1;
                inner.snapshot.state = MediaPlaybackState::Playing;
                inner.snapshot.state_revision =
                    Some(inner.snapshot.state_revision.unwrap_or(0) + 1);
                Ok(inner.snapshot)
            })
        }
    }

    #[test]
    fn playing_media_is_paused_and_resumed_by_its_session() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 0));

        controller.finish_session(session);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 1));
    }

    #[test]
    fn already_paused_media_is_left_unchanged() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Paused));
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        controller.finish_session(session);
        controller.flush_for_test();

        assert_eq!(backend.counts(), (0, 0));
    }

    #[test]
    fn cancellation_resumes_only_a_controller_owned_pause() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        controller.cancel_session(session);
        controller.flush_for_test();

        assert_eq!(backend.counts(), (1, 1));
    }

    #[test]
    fn overlapping_session_takes_pause_ownership_from_older_generation() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let first = controller.begin_session();
        controller.flush_for_test();
        let second = controller.begin_session();
        controller.flush_for_test();

        controller.finish_session(first);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 0));

        controller.finish_session(second);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 1));
    }

    #[test]
    fn ended_overlap_fences_older_owner_before_its_end_arrives() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let first = controller.begin_session();
        controller.flush_for_test();

        let second = controller.begin_session();
        controller.finish_session(second);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 1));

        controller.finish_session(first);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 1));
    }

    #[test]
    fn manual_playback_state_revision_change_prevents_resume() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        backend.manual_state_change(MediaPlaybackState::Paused);

        controller.finish_session(session);
        controller.flush_for_test();

        assert_eq!(backend.counts(), (1, 0));
    }

    #[test]
    fn manual_stop_prevents_resume_even_without_needing_revision_detection() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        backend.manual_state_change(MediaPlaybackState::Stopped);

        controller.finish_session(session);
        controller.flush_for_test();

        assert_eq!(backend.counts(), (1, 0));
    }

    #[test]
    fn unavailable_media_service_is_non_fatal() {
        let backend = Arc::new(MockBackend::unavailable());
        let controller = MediaSessionController::new(backend.clone());

        let session = controller.begin_session();
        controller.flush_for_test();
        controller.cancel_session(session);
        controller.flush_for_test();

        assert_eq!(backend.counts(), (0, 0));
    }

    #[test]
    fn media_call_timeout_is_bounded_off_the_caller_path() {
        let backend = Arc::new(
            MockBackend::new(MediaPlaybackState::Playing).with_delay(Duration::from_secs(1)),
        );
        let controller =
            MediaSessionController::with_timeout(backend.clone(), Duration::from_millis(20));

        let begin_started = Instant::now();
        let session = controller.begin_session();
        assert!(begin_started.elapsed() < Duration::from_millis(20));

        let worker_started = Instant::now();
        controller.flush_for_test();
        assert!(worker_started.elapsed() < Duration::from_millis(250));
        assert_eq!(backend.counts(), (0, 0));

        controller.cancel_session(session);
        controller.flush_for_test();
    }
}
