use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Mutex,
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

/// Construct the app-facing controller with the best native adapter available
/// on this platform. Unsupported platforms deliberately fail open through an
/// unavailable backend rather than changing recording behavior.
pub fn system_recording_media_controller() -> RecordingMediaController {
    RecordingMediaController::new(MediaSessionController::new(system_media_backend()))
}

fn system_media_backend() -> Arc<dyn MediaBackend> {
    #[cfg(target_os = "linux")]
    {
        return Arc::new(MprisMediaBackend::default());
    }

    #[cfg(not(target_os = "linux"))]
    {
        Arc::new(UnavailableMediaBackend)
    }
}

#[cfg(not(target_os = "linux"))]
struct UnavailableMediaBackend;

#[cfg(not(target_os = "linux"))]
impl MediaBackend for UnavailableMediaBackend {
    fn snapshot(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async { Err(MediaControlError::Unavailable) })
    }

    fn pause(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async { Err(MediaControlError::Unavailable) })
    }

    fn resume(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async { Err(MediaControlError::Unavailable) })
    }
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct MprisMediaBackend {
    selected_player: Mutex<Option<String>>,
}

#[cfg(target_os = "linux")]
impl MprisMediaBackend {
    async fn connection() -> Result<zbus::Connection, MediaControlError> {
        zbus::Connection::session()
            .await
            .map_err(|error| MediaControlError::Failed(error.to_string()))
    }

    async fn list_players(connection: &zbus::Connection) -> Result<Vec<String>, MediaControlError> {
        let proxy = zbus::Proxy::new(
            connection,
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
        )
        .await
        .map_err(|error| MediaControlError::Failed(error.to_string()))?;
        let names: Vec<zbus::names::OwnedBusName> = proxy
            .call("ListNames", &())
            .await
            .map_err(|error| MediaControlError::Failed(error.to_string()))?;
        Ok(names
            .into_iter()
            .map(|name| name.to_string())
            .filter(|name| name.starts_with("org.mpris.MediaPlayer2."))
            .collect())
    }

    async fn player_proxy<'a>(
        connection: &'a zbus::Connection,
        player: &'a str,
    ) -> Result<zbus::Proxy<'a>, MediaControlError> {
        zbus::Proxy::new(
            connection,
            player,
            "/org/mpris/MediaPlayer2",
            "org.mpris.MediaPlayer2.Player",
        )
        .await
        .map_err(|error| MediaControlError::Failed(error.to_string()))
    }

    async fn player_snapshot(
        connection: &zbus::Connection,
        player: &str,
    ) -> Result<MediaSnapshot, MediaControlError> {
        let proxy = Self::player_proxy(connection, player).await?;
        let status: String = proxy
            .get_property("PlaybackStatus")
            .await
            .map_err(|error| MediaControlError::Failed(error.to_string()))?;
        let state = match status.as_str() {
            "Playing" => MediaPlaybackState::Playing,
            "Paused" => MediaPlaybackState::Paused,
            "Stopped" => MediaPlaybackState::Stopped,
            _ => MediaPlaybackState::Unknown,
        };
        let state_revision = proxy
            .get_property::<i64>("Position")
            .await
            .ok()
            .and_then(|position| u64::try_from(position).ok());

        Ok(MediaSnapshot {
            state,
            session_key: Some(opaque_player_key(player)),
            state_revision,
        })
    }

    fn selected_player(&self) -> Option<String> {
        self.selected_player
            .lock()
            .expect("MPRIS player mutex poisoned")
            .clone()
    }

    fn set_selected_player(&self, player: Option<String>) {
        *self
            .selected_player
            .lock()
            .expect("MPRIS player mutex poisoned") = player;
    }

    async fn snapshot_current(&self) -> Result<MediaSnapshot, MediaControlError> {
        let connection = Self::connection().await?;
        let players = Self::list_players(&connection).await?;
        let selected = self.selected_player();
        let mut selected_snapshot = None;

        // Prefer any player that is actively playing. That lets a newly-active
        // player supersede a stale paused selection while still preserving the
        // selected paused player for the resume check when nothing else is active.
        for player in players {
            let Ok(snapshot) = Self::player_snapshot(&connection, &player).await else {
                continue;
            };
            if snapshot.state == MediaPlaybackState::Playing {
                self.set_selected_player(Some(player));
                return Ok(snapshot);
            }
            if selected.as_deref() == Some(player.as_str()) {
                selected_snapshot = Some(snapshot);
            }
        }

        if let Some(snapshot) = selected_snapshot {
            return Ok(snapshot);
        }

        self.set_selected_player(None);
        Ok(MediaSnapshot::state_only(MediaPlaybackState::Unknown))
    }

    async fn send_selected_command(
        &self,
        method: &str,
    ) -> Result<MediaSnapshot, MediaControlError> {
        let player = self
            .selected_player()
            .ok_or(MediaControlError::Unavailable)?;
        let connection = Self::connection().await?;
        let proxy = Self::player_proxy(&connection, &player).await?;
        let _: () = proxy
            .call(method, &())
            .await
            .map_err(|error| MediaControlError::Failed(error.to_string()))?;
        Self::player_snapshot(&connection, &player).await
    }
}

#[cfg(target_os = "linux")]
impl MediaBackend for MprisMediaBackend {
    fn snapshot(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async move { self.snapshot_current().await })
    }

    fn pause(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async move { self.send_selected_command("Pause").await })
    }

    fn resume(&self) -> MediaFuture<'_, MediaSnapshot> {
        Box::pin(async move {
            let result = self.send_selected_command("Play").await;
            if result.is_ok() {
                self.set_selected_player(None);
            }
            result
        })
    }
}

#[cfg(target_os = "linux")]
fn opaque_player_key(player: &str) -> u64 {
    // FNV-1a is sufficient here: this is only an in-memory equality token and
    // never leaves the controller or enters diagnostics/history.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in player.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

/// App-facing recording lifecycle wrapper around the asynchronous media worker.
///
/// The active token is kept separately from the controller's generation ledger so
/// callers never block on platform media APIs. Disabled sessions are a true no-op.
#[derive(Clone)]
pub struct RecordingMediaController {
    controller: MediaSessionController,
    active_session: Arc<Mutex<Option<MediaSession>>>,
}

impl RecordingMediaController {
    pub fn new(controller: MediaSessionController) -> Self {
        Self {
            controller,
            active_session: Arc::new(Mutex::new(None)),
        }
    }

    pub fn begin_recording(&self, enabled: bool) {
        if !enabled {
            log_media_diagnostic("begin", "disabled", None);
            return;
        }

        let session = self.controller.begin_session();
        let previous = self
            .active_session
            .lock()
            .expect("recording media session mutex poisoned")
            .replace(session);

        // A second recording generation can supersede an older token before its
        // stop reaches us. Queueing the stale end is safe: the controller fences
        // it against the newer active generation and transfers pause ownership.
        if let Some(previous) = previous {
            self.controller.finish_session(previous);
        }
        log_media_diagnostic("begin", "queued", Some(session.generation()));
    }

    pub fn finish_recording(&self) {
        let session = self
            .active_session
            .lock()
            .expect("recording media session mutex poisoned")
            .take();
        if let Some(session) = session {
            self.controller.finish_session(session);
            log_media_diagnostic("finish", "queued", Some(session.generation()));
        }
    }

    pub fn cancel_recording(&self) {
        let session = self
            .active_session
            .lock()
            .expect("recording media session mutex poisoned")
            .take();
        if let Some(session) = session {
            self.controller.cancel_session(session);
            log_media_diagnostic("cancel", "queued", Some(session.generation()));
        }
    }
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
            log_nonfatal("inspect_before_pause", generation, &error);
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
        Ok(_) => {
            log_media_diagnostic("pause", "unexpected_state", Some(generation));
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
    log_media_diagnostic("pause", "success", Some(active_now));
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
            log_nonfatal("inspect_before_resume", generation, &error);
            return;
        }
    };

    if current.state != MediaPlaybackState::Paused || media_state_was_changed(paused, current) {
        log_media_diagnostic("resume", "skipped_state_changed", Some(generation));
        return;
    }

    if let Err(error) = call_with_timeout(timeout, backend.resume()).await {
        log_nonfatal("resume", generation, &error);
    } else {
        log_media_diagnostic("resume", "success", Some(generation));
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

fn diagnostic_failure_outcome(error: &MediaControlError) -> &'static str {
    match error {
        MediaControlError::Unavailable => "unavailable",
        MediaControlError::Timeout => "timeout",
        MediaControlError::Failed(_) => "failed",
    }
}

fn log_nonfatal(action: &str, generation: u64, error: &MediaControlError) {
    let outcome = diagnostic_failure_outcome(error);
    // Intentionally categorical: backend error strings may contain player/app
    // details. Diagnostics retain only action/outcome/generation metadata.
    log_media_diagnostic(action, outcome, Some(generation));
}

fn log_media_diagnostic(action: &str, outcome: &str, generation: Option<u64>) {
    if let Some(generation) = generation {
        log::debug!("media_control action={action} outcome={outcome} generation={generation}");
    } else {
        log::debug!("media_control action={action} outcome={outcome}");
    }
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
    fn disabled_recording_media_control_is_a_true_noop() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());
        let recording_media = RecordingMediaController::new(controller.clone());

        recording_media.begin_recording(false);
        recording_media.finish_recording();
        controller.flush_for_test();

        assert_eq!(backend.counts(), (0, 0));
    }

    #[test]
    fn enabled_recording_media_control_uses_async_controller_lifecycle() {
        let backend = Arc::new(MockBackend::new(MediaPlaybackState::Playing));
        let controller = MediaSessionController::new(backend.clone());
        let recording_media = RecordingMediaController::new(controller.clone());

        recording_media.begin_recording(true);
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 0));

        recording_media.finish_recording();
        controller.flush_for_test();
        assert_eq!(backend.counts(), (1, 1));
    }

    #[test]
    fn media_failure_diagnostics_are_categorical_and_drop_backend_detail() {
        let error = MediaControlError::Failed("private player title or service detail".to_string());
        assert_eq!(diagnostic_failure_outcome(&error), "failed");
        assert!(!diagnostic_failure_outcome(&error).contains("player"));
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
