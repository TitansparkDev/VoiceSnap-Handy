use crate::audio_toolkit::{
    apply_vocabulary_corrections, build_vocabulary_prompt, detect_output_language,
    normalize_transcription_output, remove_filler_words, OutputLanguageEvidence,
};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::{EngineType, ModelManager};
use crate::settings::{
    get_settings, AppSettings, InsertionMode, ModelUnloadTimeout, OrtAcceleratorSetting,
    TranscribeAcceleratorSetting,
};
use crate::transcription_coordinator::{
    LiveInsertionAttempt, LiveInsertionLedger, LiveInsertionOutcome,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use tauri_specta::Event;
use transcribe_cpp::{
    Backend, Feature, Model, ModelOptions, RunExtension, RunOptions, Session, StreamOptions, Task,
    WhisperRunOptions,
};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        cohere::CohereModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    SpeechModel, TranscribeOptions,
};

const STREAM_PERF_LOG_INTERVAL: Duration = Duration::from_secs(5);
const STREAM_FINALIZE_REPLY_TIMEOUT: Duration = Duration::from_secs(30);
const STREAM_WORKER_QUIESCE_TIMEOUT: Duration = Duration::from_secs(2);

fn is_model_switch(current_model: Option<&str>, requested_model: &str) -> bool {
    current_model.is_some_and(|current| current != requested_model)
}

fn clear_live_insertion_state(
    active: &mut Option<LiveInsertionLedger>,
    blocked_after_clear: &AtomicBool,
) {
    if active
        .as_ref()
        .is_some_and(LiveInsertionLedger::blocks_final_paste)
    {
        blocked_after_clear.store(true, Ordering::Release);
    }
    *active = None;
}

fn terminate_live_insertion_state(
    active: &mut Option<LiveInsertionLedger>,
    blocked_after_clear: &AtomicBool,
) {
    if let Some(ledger) = active.as_mut() {
        ledger.cancel();
    }
    clear_live_insertion_state(active, blocked_after_clear);
}

fn clear_terminal_live_insertion_state(
    active: &mut Option<LiveInsertionLedger>,
    blocked_after_clear: &AtomicBool,
) {
    if active
        .as_ref()
        .is_some_and(|ledger| ledger.stop_reason().is_some())
    {
        clear_live_insertion_state(active, blocked_after_clear);
    }
}

fn live_insertion_state_blocks_final_paste(
    active: Option<&LiveInsertionLedger>,
    blocked_after_clear: bool,
) -> bool {
    blocked_after_clear || active.is_some_and(LiveInsertionLedger::blocks_final_paste)
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

/// Live transcription snapshot emitted to the overlay during a streaming run.
/// `committed` is the append-only, flicker-free prefix; `tentative` is the
/// volatile suffix the model may still rewrite.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamTextEvent {
    pub committed: String,
    pub tentative: String,
}

impl StreamTextEvent {
    /// The only stream text eligible for experimental external insertion.
    /// Keeping this accessor committed-only makes it impossible for the volatile
    /// tentative suffix to enter the live insertion adapter accidentally.
    #[allow(dead_code)] // steward hook: final Wave 3 settings/action wiring activates this
    pub(crate) fn committed_for_live_insertion(&self) -> &str {
        &self.committed
    }
}

/// Phase of the streaming overlay card, emitted to drive its UI state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamPhase {
    /// Receiving audio / live text (or waiting for the stream to begin). Rust
    /// does not emit this today; the frontend starts in this phase and Rust only
    /// emits transitions away from it.
    Listening,
    /// Finalizing or post-processing — show a spinner.
    Working,
}

/// Semantic kind of "working" phase, used to localize the spinner label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamWorkKind {
    Transcribing,
    Polishing,
}

/// Emitted to switch the streaming overlay to a working spinner.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamPhaseEvent {
    pub phase: StreamPhase,
    /// Present only when `phase` is `Working`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<StreamWorkKind>,
}

/// Commands sent to the streaming worker thread. Audio frames and the finalize
/// request travel the same channel so FIFO ordering guarantees every fed frame
/// is processed before finalize runs.
enum StreamCmd {
    Feed(Vec<f32>),
    /// Flush the stream and reply with the final text, or `None` if no stream
    /// was ever active (caller should fall back to batch transcription).
    Finalize(mpsc::Sender<Option<FinalizedStreamText>>),
    Cancel,
}

struct FinalizedStreamText {
    text: String,
    output_language: OutputLanguageEvidence,
    /// The streaming model's supported languages, for text-based detection.
    supported_languages: Vec<String>,
    benchmark_timing: StreamBenchmarkTiming,
}

/// Safe timing-only sample for the fixed-WAV stream benchmark. This structure
/// intentionally contains no transcript, audio, clipboard, window, or path data.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct StreamBenchmarkTiming {
    pub first_partial_ms: Option<u64>,
    pub committed_cadence_ms: Vec<u64>,
    pub finalization_tail_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TranscriptionBenchmarkSample {
    pub mode: String,
    pub audio_ms: u64,
    pub first_partial_ms: Option<u64>,
    pub committed_cadence_ms: Vec<u64>,
    pub finalization_tail_ms: u64,
    pub total_ms: u64,
    pub worker_released: bool,
    /// Deterministic word-error rate in thousandths (0 = exact, 1000 = 100%).
    /// None when the benchmark did not receive a reference transcript.
    pub word_error_rate_milli: Option<u32>,
}

fn benchmark_word_error_rate_milli(reference: &str, actual: &str) -> u32 {
    fn words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|ch| ch.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    let reference = words(reference);
    let actual = words(actual);
    if reference.is_empty() {
        return if actual.is_empty() { 0 } else { 1000 };
    }

    let mut previous: Vec<usize> = (0..=actual.len()).collect();
    for (row, expected) in reference.iter().enumerate() {
        let mut current = Vec::with_capacity(actual.len() + 1);
        current.push(row + 1);
        for (column, observed) in actual.iter().enumerate() {
            let substitution = previous[column] + usize::from(expected != observed);
            let insertion = current[column] + 1;
            let deletion = previous[column + 1] + 1;
            current.push(substitution.min(insertion).min(deletion));
        }
        previous = current;
    }
    let edits = previous[actual.len()] as u64;
    ((edits * 1000) / reference.len() as u64).min(u32::MAX as u64) as u32
}

/// Routes real-time audio frames to the active streaming worker. Shared between
/// the [`TranscriptionManager`] (opens/closes the route) and the audio recorder's
/// per-frame callback (feeds frames). The recorder holds an `Arc<StreamRouter>`
/// directly, so a frame with no stream pending costs a single relaxed atomic
/// load — no Tauri state lookup, no mutex lock.
pub struct StreamRouter {
    /// Command channel to the active streaming worker, present from
    /// `start_stream` until `finalize_stream`/`cancel_stream`.
    tx: Mutex<Option<mpsc::Sender<StreamCmd>>>,
    /// True while a stream is pending or active (channel is open). The audio
    /// callback checks this first to avoid the mutex lock when no stream runs.
    open: Arc<AtomicBool>,
}

impl StreamRouter {
    fn new() -> Self {
        Self {
            tx: Mutex::new(None),
            open: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Open a fresh command channel for a new streaming session, returning the
    /// receiver the worker should drain. Caller must ensure no prior channel is
    /// still open.
    fn open(&self) -> mpsc::Receiver<StreamCmd> {
        let (tx, rx) = mpsc::channel::<StreamCmd>();
        *self.tx.lock().unwrap() = Some(tx);
        self.open.store(true, Ordering::Relaxed);
        rx
    }

    /// Take the sender out (closing the channel to new feeds). Returns the
    /// sender so the caller can send the final `Finalize`/`Cancel` command.
    fn take(&self) -> Option<mpsc::Sender<StreamCmd>> {
        self.open.store(false, Ordering::Relaxed);
        self.tx.lock().unwrap().take()
    }

    /// Drop the channel and mark closed without sending a final command (used
    /// when the worker exits without a finalize/cancel handshake).
    fn clear(&self) {
        self.open.store(false, Ordering::Relaxed);
        *self.tx.lock().unwrap() = None;
    }

    /// Forward a 16 kHz frame to the active streaming worker. Cheap no-op (a
    /// single relaxed atomic load) when no stream is pending.
    pub fn feed(&self, frame: &[f32]) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(StreamCmd::Feed(frame.to_vec()));
        }
    }

    /// Whether a stream is pending or active.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribeSelectionPlanMetadata {
    /// Persisted accelerator intent used for this load. `None` denotes a one-off
    /// command-line device override rather than a saved app preference.
    pub saved_accelerator: Option<String>,
    /// Stable persisted GPU identity when the saved preference pinned one.
    pub saved_gpu_device: Option<String>,
    /// Backend requested by Handy before the runtime could fall back.
    pub recommended_backend: String,
    /// Readable exact device Handy recommended, when one was pinned.
    pub recommended_device: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribeRuntimeMetadata {
    /// Backend the loaded transcribe.cpp model was actually bound to.
    pub backend: String,
    /// Readable device the loaded transcribe.cpp model actually used.
    pub device: Option<String>,
    /// Stable recovery cause when acceleration was downgraded for this run.
    pub recovery_reason: Option<String>,
}

fn transcribe_runtime_metadata(
    backend: String,
    device: Option<String>,
    recovery_reason: Option<String>,
) -> TranscribeRuntimeMetadata {
    TranscribeRuntimeMetadata {
        backend,
        device,
        recovery_reason,
    }
}

enum LoadedEngine {
    /// Whisper-family models (whisper, breeze-asr, custom .bin/.gguf) via
    /// transcribe-cpp. Holds the live `Session`, which keeps its `Model` alive
    /// internally, so repeated dictation reuses the session without reloading.
    TranscribeCpp(Session),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    Cohere(CohereModel),
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        // Recover from a poisoned mutex instead of panicking —
        // a panic inside Drop calls abort().
        let mut is_loading = match self.is_loading.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned is_loading mutex during LoadingGuard drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

/// RAII guard that clears the streaming worker/lease flags on any worker exit -
/// normal return, early return, or a panic in an engine call that unwinds the
/// detached worker thread. Tokens prevent an older worker from clearing a newer
/// worker's state if a start/finalize race ever slips through.
struct StreamWorkerGuard {
    worker_id: u64,
    active_stream_worker: Arc<AtomicU64>,
    active_engine_lease: Arc<AtomicU64>,
    stream_active: Arc<AtomicBool>,
}

impl Drop for StreamWorkerGuard {
    fn drop(&mut self) {
        if self.active_stream_worker.load(Ordering::Acquire) == self.worker_id {
            self.stream_active.store(false, Ordering::Release);
        }
        let _ = self.active_engine_lease.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.active_stream_worker.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
    reload_model_on_next_use: Arc<AtomicBool>,
    /// Routes real-time audio frames to the active streaming worker; see
    /// [`StreamRouter`]. Shared with the audio recorder so per-frame feeds skip
    /// Tauri state and the manager lock.
    router: Arc<StreamRouter>,
    /// True only while a transcribe-cpp `Stream` is actually in flight (set by
    /// the worker once `stream()` succeeds). Used for overlay/UI decisions.
    stream_active: Arc<AtomicBool>,
    /// Streaming uses four independent flags: router open = frames should route,
    /// worker active = no second worker may start, engine lease = engine is out
    /// of the mutex, stream active = UI should show a live session.
    ///
    /// Monotonic id source for stream workers; zero means "no worker".
    next_stream_worker_id: Arc<AtomicU64>,
    /// Nonzero while a stream worker exists, even if it has not leased the engine
    /// yet. This prevents a second worker from starting after finalize/cancel
    /// closes the router but before the first worker has fully exited.
    active_stream_worker: Arc<AtomicU64>,
    /// Nonzero while the streaming worker has taken the engine out of `engine`.
    /// `is_model_loaded()` consults this so the model still reports "loaded"
    /// while the worker holds it.
    active_engine_lease: Arc<AtomicU64>,
    /// Plan-time accelerator metadata for the most recent transcribe.cpp load.
    /// Kept separate from runtime truth so persisted intent and recommendation do
    /// not get rewritten when the loaded backend falls back.
    selection_plan: Arc<Mutex<Option<TranscribeSelectionPlanMetadata>>>,
    /// Actual backend/device bound by the most recent transcribe.cpp load. This
    /// survives an unhealthy engine being dropped so failed sessions still retain
    /// truthful runtime diagnostics.
    runtime_metadata: Arc<Mutex<Option<TranscribeRuntimeMetadata>>>,
    /// Process-local health latch. Once an accelerated transcribe.cpp session
    /// fails during inference, later persisted loads use CPU for this app run
    /// without rewriting the user's saved accelerator preference.
    force_cpu_for_run: Arc<AtomicBool>,
    /// Monotonic token source for committed-only insertion sessions. Attempt
    /// sequence numbers restart per recording, so queued native-input work also
    /// carries this token to reject stale work after cancellation/model teardown.
    next_live_insertion_session_id: Arc<AtomicU64>,
    /// Optional committed-only insertion ledger for the currently active session.
    live_insertion: Arc<Mutex<Option<LiveInsertionLedger>>>,
    /// Sticky within one recording session. A model unload/cancel/engine failure
    /// may need to clear the active ledger before the action reaches final-output
    /// handling; remember whether that cleared ledger had already inserted text
    /// or crossed a safety boundary so a whole-transcript paste cannot duplicate
    /// or retarget it. Reset only when the next insertion session begins.
    live_insertion_blocks_final_paste_after_clear: Arc<AtomicBool>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
            reload_model_on_next_use: Arc::new(AtomicBool::new(false)),
            router: Arc::new(StreamRouter::new()),
            stream_active: Arc::new(AtomicBool::new(false)),
            next_stream_worker_id: Arc::new(AtomicU64::new(1)),
            active_stream_worker: Arc::new(AtomicU64::new(0)),
            active_engine_lease: Arc::new(AtomicU64::new(0)),
            selection_plan: Arc::new(Mutex::new(None)),
            runtime_metadata: Arc::new(Mutex::new(None)),
            force_cpu_for_run: Arc::new(AtomicBool::new(false)),
            next_live_insertion_session_id: Arc::new(AtomicU64::new(1)),
            live_insertion: Arc::new(Mutex::new(None)),
            live_insertion_blocks_final_paste_after_clear: Arc::new(AtomicBool::new(false)),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .is_some_and(|a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        // The engine may be leased out to the streaming worker (taken out of
        // the mutex). It's still loaded, just in use, so report true.
        self.lock_engine().is_some() || self.active_engine_lease.load(Ordering::Acquire) != 0
    }

    /// Accelerator changes should not disturb the current transcription. Mark
    /// the cached engine stale; the next model-use path reloads it with the
    /// latest settings.
    pub fn reload_model_on_next_use(&self) {
        self.reload_model_on_next_use.store(true, Ordering::Release);
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        // Model unload is a terminal lifecycle boundary for experimental live
        // insertion. Quiesce the worker first so no late committed event can
        // race the cleared session or retain the engine lease.
        self.cancel_stream();
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout != ModelUnloadTimeout::Immediately
            || !self.is_model_loaded()
        {
            return;
        }

        // Keep a live session's ledger until the action has made its final-paste
        // decision. Manual unload still terminates it immediately via
        // `unload_model`; this only defers the automatic post-transcription
        // cleanup by the short remainder of the output pipeline.
        let live_session_pending = self
            .live_insertion
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|ledger| ledger.mode() == InsertionMode::LiveCommittedExperimental);
        if live_session_pending {
            debug!("Deferring immediate model unload until live insertion session is closed");
            return;
        }

        info!("Immediately unloading model after {}", context);
        if let Err(e) = self.unload_model() {
            warn!("Failed to immediately unload model: {}", e);
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        self.load_model_with_device(model_id, None)
    }

    /// Like [`load_model`](Self::load_model), but lets a caller hard-select the
    /// compute device for this one load by its `transcribe_cpp::devices()`
    /// registry index (the index shown by `--list-devices`). `None` keeps the
    /// persisted accelerator setting (which may be Auto). Only affects
    /// transcribe-cpp (whisper-family) models; the selection is not persisted.
    pub fn load_model_with_device(
        &self,
        model_id: &str,
        device_index: Option<usize>,
    ) -> Result<()> {
        let switching_model = is_model_switch(
            self.current_model_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_deref(),
            model_id,
        );
        if switching_model {
            // A real model switch permanently ends the previous session's
            // insertion contract and must not race its leased streaming engine.
            self.cancel_live_insertion();
            if !self.quiesce_stream_worker(STREAM_WORKER_QUIESCE_TIMEOUT) {
                self.clear_live_insertion();
                return Err(anyhow::anyhow!(
                    "Timed out waiting {:?} for the previous streaming worker before loading model '{}'",
                    STREAM_WORKER_QUIESCE_TIMEOUT,
                    model_id
                ));
            }
            self.clear_live_insertion();
        }
        // Initial lazy loading is deliberately not treated as a switch: a newly
        // opened streaming worker may already be waiting on this same load.

        apply_accelerator_settings(&self.app_handle);

        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = self.model_manager.get_model_path(model_id)?;

        // Drop the current engine BEFORE building the new one so transcribe-cpp
        // frees the previous native context first — avoids holding two models at
        // once (peak memory on large GGUFs). Clear the id too: if the new load
        // fails, status should read "no loaded model", not the dropped engine.
        {
            let mut engine = self.lock_engine();
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }
        // A new load supersedes the previous planning/runtime metadata. Non-
        // transcribe engines intentionally leave these empty.
        *self.selection_plan.lock().unwrap() = None;
        *self.runtime_metadata.lock().unwrap() = None;

        // Create appropriate engine based on model type
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        let loaded_engine = match model_info.engine_type {
            EngineType::TranscribeCpp => {
                // The whisper backend is chosen at load time (transcribe-cpp has
                // no runtime global). With an explicit `device_index` (the
                // --device-index flag) hard-select that registered device;
                // otherwise re-read the persisted accelerator preference (so an
                // accelerator change marked for reload takes effect here).
                let (backend, device, selection_plan, planned_recovery_reason) = match device_index
                {
                    Some(index) => {
                        let (backend, device) = resolve_device_index(index).inspect_err(|e| {
                            emit_loading_failed(&e.to_string());
                        })?;
                        let recommended_device = device.as_ref().map(transcribe_device_label);
                        (
                            backend,
                            device,
                            TranscribeSelectionPlanMetadata {
                                saved_accelerator: None,
                                saved_gpu_device: None,
                                recommended_backend: transcribe_backend_plan_label(backend)
                                    .to_string(),
                                recommended_device,
                            },
                            None,
                        )
                    }
                    None => {
                        let settings = get_settings(&self.app_handle);
                        let (recommended_backend, recommended_device) =
                            resolve_transcribe_load_plan(&settings);
                        let (saved_accelerator, saved_gpu_device) =
                            describe_saved_transcribe_preference(&settings);
                        let recommended_device_label =
                            recommended_device.as_ref().map(transcribe_device_label);
                        let force_cpu = should_force_transcribe_cpu_for_run(
                            self.force_cpu_for_run.load(Ordering::Acquire),
                            recommended_backend,
                        );
                        let (backend, device) = if force_cpu {
                            warn!(
                                "Previous accelerated transcription failed during this app run; loading '{}' on CPU without changing the saved accelerator preference",
                                model_id
                            );
                            (Backend::Cpu, None)
                        } else {
                            (recommended_backend, recommended_device)
                        };
                        (
                            backend,
                            device,
                            TranscribeSelectionPlanMetadata {
                                saved_accelerator: Some(saved_accelerator),
                                saved_gpu_device,
                                recommended_backend: transcribe_backend_plan_label(
                                    recommended_backend,
                                )
                                .to_string(),
                                recommended_device: recommended_device_label,
                            },
                            force_cpu.then(|| "runtime_health_fallback".to_string()),
                        )
                    }
                };
                *self.selection_plan.lock().unwrap() = Some(selection_plan);
                let requested_device = device
                    .as_ref()
                    .map(transcribe_device_label)
                    .unwrap_or_else(|| "automatic".to_string());
                let allow_cpu_fallback = should_retry_transcribe_load_on_cpu(device_index, backend);
                let model_options = ModelOptions { backend, device };
                let (model, recovery_reason) = match Model::load_with(&model_path, &model_options) {
                    Ok(model) => (model, planned_recovery_reason),
                    Err(primary_err) if allow_cpu_fallback => {
                        warn!(
                            "Failed to load whisper model '{}' with requested backend {:?} and device '{}': {}; retrying on CPU for this run without changing the saved accelerator preference",
                            model_id, backend, requested_device, primary_err
                        );
                        let cpu_options = ModelOptions {
                            backend: Backend::Cpu,
                            device: None,
                        };
                        let model = Model::load_with(&model_path, &cpu_options).map_err(|cpu_err| {
                            let error_msg = format!(
                                "Failed to load whisper model {} with requested acceleration ({}), then CPU fallback also failed: {}",
                                model_id, primary_err, cpu_err
                            );
                            emit_loading_failed(&error_msg);
                            anyhow::anyhow!(error_msg)
                        })?;
                        (model, Some("startup_gpu_fallback".to_string()))
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        return Err(anyhow::anyhow!(error_msg));
                    }
                };
                // The bound backend may differ from the request (e.g. CPU
                // fallback under Auto). Snapshot it before session creation so a
                // later runtime-health failure cannot erase the actual device used.
                let bound_backend = model.backend().to_string();
                let bound_device = model
                    .device()
                    .ok()
                    .map(|device| transcribe_device_label(&device));
                *self.runtime_metadata.lock().unwrap() = Some(transcribe_runtime_metadata(
                    bound_backend.clone(),
                    bound_device.clone(),
                    recovery_reason,
                ));
                let session = model.session().map_err(|e| {
                    let error_msg = format!(
                        "Failed to create session for whisper model {}: {}",
                        model_id, e
                    );
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                // Reconcile the registry's advertised capabilities with the
                // loaded model's real ones (GGUF metadata) so badges/gating
                // reflect runtime truth, not the pre-download probe. The
                // load-completed event below triggers the frontend refresh.
                let caps = session.model().capabilities();
                self.model_manager.set_runtime_capabilities(
                    model_id,
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect,
                    caps.languages.clone(),
                );
                info!(
                    "Loaded whisper model '{}' (requested {:?}, requested device '{}', \
                     bound backend '{}', bound device '{}', supports_streaming={}, \
                     supports_translate={}, supports_language_detect={})",
                    model_id,
                    backend,
                    requested_device,
                    bound_backend,
                    bound_device.as_deref().unwrap_or("unknown"),
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect
                );
                LoadedEngine::TranscribeCpp(session)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::Cohere => {
                let engine = CohereModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load cohere model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Cohere(engine)
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return;
        }

        let reload_pending = self.reload_model_on_next_use.load(Ordering::Acquire);
        if !reload_pending && self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            if reload_pending {
                self_clone
                    .reload_model_on_next_use
                    .store(false, Ordering::Release);
            }
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    /// The compute backend the currently-loaded engine is bound to, for
    /// diagnostics (e.g. confirming `--device-index` actually bound a GPU rather
    /// than falling back to CPU/auto). transcribe-cpp (whisper-family) reports
    /// its real backend string; ONNX engines report "onnx"; `None` when no
    /// model is loaded.
    pub fn current_backend(&self) -> Option<String> {
        match self.lock_engine().as_ref() {
            Some(LoadedEngine::TranscribeCpp(session)) => {
                Some(session.model().backend().to_string())
            }
            Some(_) => Some("onnx".to_string()),
            None => None,
        }
    }

    /// The actual compute device the currently-loaded engine is using. Whisper-
    /// family engines report the device bound by transcribe-cpp; the ONNX engines
    /// in this build are CPU-only and therefore report `cpu`.
    pub fn current_device(&self) -> Option<String> {
        match self.lock_engine().as_ref() {
            Some(LoadedEngine::TranscribeCpp(session)) => session
                .model()
                .device()
                .ok()
                .map(|device| transcribe_device_label(&device)),
            Some(_) => Some("cpu".to_string()),
            None => None,
        }
    }

    /// Plan used by the most recent transcribe.cpp load. It intentionally
    /// survives model unload so the just-completed session can persist it.
    pub fn selection_plan_metadata(&self) -> Option<TranscribeSelectionPlanMetadata> {
        self.selection_plan.lock().unwrap().clone()
    }

    /// Runtime truth for history/diagnostics. transcribe.cpp snapshots survive an
    /// unhealthy engine being dropped; other engines fall back to the live engine
    /// getters because they do not use the transcribe.cpp recovery path.
    pub fn runtime_metadata(&self) -> (Option<String>, Option<String>, Option<String>) {
        if let Some(metadata) = self.runtime_metadata.lock().unwrap().clone() {
            return (
                Some(metadata.backend),
                metadata.device,
                metadata.recovery_reason,
            );
        }
        (self.current_backend(), self.current_device(), None)
    }

    /// Whether a live streaming run is currently in flight.
    pub fn is_streaming(&self) -> bool {
        self.stream_active.load(Ordering::Acquire)
    }

    /// Shared handle to the stream router, used by the audio recorder to feed
    /// real-time frames without going through Tauri state on every frame.
    pub fn stream_router(&self) -> Arc<StreamRouter> {
        Arc::clone(&self.router)
    }

    /// Begin the insertion-side state for one recording session. This is an
    /// intentionally separate opt-in from streaming preview: callers must choose
    /// `LiveCommittedExperimental` explicitly, so merely using a streaming model
    /// never changes Handy's existing final-paste behavior.
    pub(crate) fn begin_insertion_session(&self, mode: InsertionMode) {
        let target = (mode == InsertionMode::LiveCommittedExperimental)
            .then(crate::paste_tx::capture_target_identity)
            .flatten();
        self.live_insertion_blocks_final_paste_after_clear
            .store(false, Ordering::Release);
        let session_id = self
            .next_live_insertion_session_id
            .fetch_add(1, Ordering::Relaxed);
        *self.live_insertion.lock().unwrap() = Some(LiveInsertionLedger::with_session_id(
            mode, target, session_id,
        ));
    }

    /// Mode fixed at session start. Callers use this for history/final-output
    /// handling so changing settings while recording cannot retroactively change
    /// the safety contract of the active session.
    pub(crate) fn current_insertion_mode(&self) -> InsertionMode {
        self.live_insertion
            .lock()
            .unwrap()
            .as_ref()
            .map(LiveInsertionLedger::mode)
            .unwrap_or_default()
    }

    /// Feed only the committed part of a stream snapshot into the live ledger.
    /// The returned attempt must be executed by the native input adapter and its
    /// exact result acknowledged with `record_live_insertion_result`.
    #[allow(dead_code)] // steward hook: native input adapter will consume returned attempts
    pub(crate) fn observe_live_stream_text(
        &self,
        event: &StreamTextEvent,
    ) -> Option<LiveInsertionAttempt> {
        let current_target = crate::paste_tx::capture_target_identity();
        let mut guard = self.live_insertion.lock().unwrap();
        let attempt = guard.as_mut().and_then(|ledger| {
            ledger.observe_committed(
                event.committed_for_live_insertion(),
                current_target.as_ref(),
            )
        });
        clear_terminal_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
        attempt
    }

    /// Positive speech evidence is supplied separately from model text. This is
    /// the latch that prevents a committed-looking silence hallucination from
    /// becoming the first external insertion. Capture foreground identity only
    /// for the first positive frame; subsequent VAD-approved frames are a cheap
    /// no-op rather than a per-frame OS focus query.
    pub(crate) fn observe_live_speech_evidence(&self) -> Option<LiveInsertionAttempt> {
        let mut guard = self.live_insertion.lock().unwrap();
        let ledger = guard.as_mut()?;
        if ledger.mode() != InsertionMode::LiveCommittedExperimental || ledger.has_speech_evidence()
        {
            return None;
        }
        let current_target = crate::paste_tx::capture_target_identity();
        let attempt = ledger.observe_speech_evidence(current_target.as_ref());
        clear_terminal_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
        attempt
    }

    /// Called by the recorder after the active VAD policy admits a speech frame.
    /// If a model emitted committed text before the latch, this executes that
    /// pending append-only delta immediately after the latch turns positive.
    pub(crate) fn observe_and_execute_live_speech_evidence(&self) {
        if let Some(attempt) = self.observe_live_speech_evidence() {
            self.execute_live_insertion_attempt(attempt);
        }
    }

    pub(crate) fn record_live_insertion_result(
        &self,
        session_id: u64,
        sequence: u64,
        outcome: LiveInsertionOutcome,
    ) -> bool {
        let mut guard = self.live_insertion.lock().unwrap();
        let recorded = guard
            .as_mut()
            .is_some_and(|ledger| ledger.record_attempt_result(session_id, sequence, outcome));
        clear_terminal_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
        recorded
    }

    /// Ask the live ledger for at most the still-uncommitted final raw tail.
    /// Callers must pass the raw streaming final, before whole-transcript cleanup.
    pub(crate) fn finalize_live_insertion_raw(
        &self,
        final_text: &str,
    ) -> Option<LiveInsertionAttempt> {
        let current_target = crate::paste_tx::capture_target_identity();
        let mut guard = self.live_insertion.lock().unwrap();
        let attempt = guard
            .as_mut()
            .and_then(|ledger| ledger.finalize(final_text, current_target.as_ref()));
        clear_terminal_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
        attempt
    }

    pub(crate) fn cancel_live_insertion(&self) {
        if let Some(ledger) = self.live_insertion.lock().unwrap().as_mut() {
            ledger.cancel();
        }
    }

    pub(crate) fn clear_live_insertion(&self) {
        let mut guard = self.live_insertion.lock().unwrap();
        clear_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
    }

    /// Whether the normal final paste is unsafe for the current session because
    /// some live text escaped already or a focus/input safety boundary fired.
    /// The sticky bit matters when model-unload policy clears the active ledger
    /// during stream finalization before the action reaches its paste decision.
    pub(crate) fn live_insertion_blocks_final_paste(&self) -> bool {
        let guard = self.live_insertion.lock().unwrap();
        live_insertion_state_blocks_final_paste(
            guard.as_ref(),
            self.live_insertion_blocks_final_paste_after_clear
                .load(Ordering::Acquire),
        )
    }

    /// Engine failure is terminal for committed insertion. Clear the active
    /// session immediately so no later stream event can append, while preserving
    /// any final-paste suppression state from text that may already have escaped.
    fn terminate_live_insertion_after_engine_failure(&self) {
        let mut guard = self.live_insertion.lock().unwrap();
        terminate_live_insertion_state(
            &mut guard,
            self.live_insertion_blocks_final_paste_after_clear.as_ref(),
        );
    }

    /// Execute one planned append on the main thread. The foreground identity is
    /// re-checked immediately before injection, closing the race between ledger
    /// planning and queued native input. The call waits for that exact attempt so
    /// committed-prefix accounting cannot run ahead of uncertain delivery.
    fn execute_live_insertion_attempt(&self, attempt: LiveInsertionAttempt) {
        let session_id = attempt.session_id;
        let sequence = attempt.sequence;
        let attempt_for_authorization = attempt.clone();
        let expected_target = attempt.target;
        let text = attempt.text;
        let app_handle = self.app_handle.clone();
        let app_for_input = app_handle.clone();
        let live_insertion = Arc::clone(&self.live_insertion);
        let (result_tx, result_rx) = mpsc::sync_channel(1);

        let schedule_result = app_handle.run_on_main_thread(move || {
            // Serialize authorization with cancellation/model teardown. A task
            // queued for an old recording must not become armed merely because a
            // new recording reused the same per-session sequence number.
            let guard = live_insertion.lock().unwrap();
            if !guard
                .as_ref()
                .is_some_and(|ledger| ledger.authorizes_attempt(&attempt_for_authorization))
            {
                let _ = result_tx.send(None);
                return;
            }

            let current_target = crate::paste_tx::capture_target_identity();
            let outcome = match current_target {
                None => LiveInsertionOutcome::TargetLost,
                Some(current) if current != expected_target => LiveInsertionOutcome::TargetChanged,
                Some(_) => match crate::clipboard::paste_live_delta(&text, &app_for_input) {
                    Ok(()) => LiveInsertionOutcome::Inserted,
                    Err(error) => {
                        warn!("Live committed insertion failed: {error}");
                        LiveInsertionOutcome::InputFailed
                    }
                },
            };
            drop(guard);
            let _ = result_tx.send(Some(outcome));
        });

        let outcome = match schedule_result {
            Ok(()) => result_rx.recv().ok().flatten(),
            Err(error) => {
                warn!("Failed to schedule live insertion on main thread: {error}");
                Some(LiveInsertionOutcome::InputFailed)
            }
        };
        if let Some(outcome) = outcome {
            let _ = self.record_live_insertion_result(session_id, sequence, outcome);
        }
    }

    /// Begin a live streaming transcription on the held engine's session.
    /// Audio frames pushed via [`StreamRouter::feed`] (captured directly by the
    /// audio recorder) are decoded incrementally and emitted to the overlay as
    /// [`StreamTextEvent`].
    ///
    /// Non-blocking: spawns a worker that waits for any in-progress model load,
    /// verifies the model supports streaming, then begins the stream. If the
    /// model can't stream, the worker idles until finalize/cancel and reports
    /// `None` so the caller falls back to batch transcription. Frames sent
    /// before the stream begins queue on the channel and are not lost.
    pub fn start_stream(&self) {
        if self.router.is_open() {
            warn!("start_stream called while a stream route is already open");
            return;
        }
        // Finalize/cancel closes the route before the detached worker returns
        // its engine lease. Give that bounded cleanup a chance to finish so an
        // immediately following dictation is usable instead of being dropped.
        if self.active_stream_worker.load(Ordering::Acquire) != 0
            && !self.wait_for_stream_worker_release(STREAM_WORKER_QUIESCE_TIMEOUT)
        {
            warn!("start_stream timed out waiting for the previous streaming worker to release");
            return;
        }
        let worker_id = self.next_stream_worker_id.fetch_add(1, Ordering::Relaxed);
        if self
            .active_stream_worker
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("start_stream lost a race with another stream worker");
            return;
        }
        let rx = self.router.open();
        self.stream_active.store(false, Ordering::Release);
        let requested_at = Instant::now();

        let manager = self.clone();
        thread::spawn(move || manager.run_stream_worker(rx, worker_id, requested_at));
    }

    fn run_stream_worker(
        &self,
        rx: mpsc::Receiver<StreamCmd>,
        worker_id: u64,
        requested_at: Instant,
    ) {
        let _worker = StreamWorkerGuard {
            worker_id,
            active_stream_worker: Arc::clone(&self.active_stream_worker),
            active_engine_lease: Arc::clone(&self.active_engine_lease),
            stream_active: Arc::clone(&self.stream_active),
        };

        // Wait for any in-progress model load to finish (start_stream races the
        // background load kicked off when recording starts).
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }
        }

        let model_id = self.get_current_model().unwrap_or_default();

        // Take the engine out of the mutex so we own it during streaming,
        // structurally excluding any concurrent batch transcription (which
        // transcribe-cpp's compute_lock would refuse anyway). Returned when the
        // worker exits, or dropped if the model was switched/unloaded mid-stream.
        if self
            .active_engine_lease
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("Live preview: another worker already holds the transcription engine");
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }
        let mut engine = match self.lock_engine().take() {
            Some(e) => e,
            None => {
                info!(
                    "Live preview: model '{}' was unloaded before streaming could begin; \
                     falling back to batch transcription",
                    model_id
                );
                let _ = self.active_engine_lease.compare_exchange(
                    worker_id,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                self.router.clear();
                drain_until_finalize(rx);
                return;
            }
        };

        // Only transcribe-cpp models expose streaming; ONNX engines fall back to
        // batch. The loaded session (not the ModelManager copy) is the source of
        // truth for run-path capabilities.
        let (supports_streaming, supports_translate, languages) = match &engine {
            LoadedEngine::TranscribeCpp(session) => {
                let model = session.model();
                let caps = model.capabilities();
                info!(
                    "Live preview: model '{}' arch='{}' variant='{}' supports_streaming={} \
                     supports_translate={} languages={:?}",
                    model_id,
                    model.arch(),
                    model.variant(),
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                );
                (
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                )
            }
            _ => {
                info!(
                    "Live preview: model '{}' is not a transcribe-cpp model; \
                     streaming is unavailable, using batch transcription",
                    model_id
                );
                (false, false, Vec::new())
            }
        };

        if !supports_streaming {
            self.return_engine(engine, &model_id);
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }

        // Build run options mirroring the offline transcribe-cpp path: task +
        // language gated against what the model actually advertises.
        let settings = get_settings(&self.app_handle);
        let effective_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &model_id);
        let run_plan = transcribe_cpp_run_plan(
            settings.translate_to_english,
            &effective_language,
            &languages,
            supports_translate,
        );
        let output_language = resolve_output_language_evidence(
            &settings,
            run_plan.language.as_deref(),
            &languages,
            run_plan.target_language.as_deref() == Some("en"),
        );
        let run_options = RunOptions {
            task: run_plan.task,
            language: run_plan.language,
            target_language: run_plan.target_language,
            ..Default::default()
        };

        // Run the stream on the held session. The Stream borrows the session
        // (and thus the engine) for its lifetime, so the feed/finalize loop
        // lives in a labeled block — when it exits, the borrow is released and
        // the engine can be moved into return_engine().
        let mut finalize_reply: Option<mpsc::Sender<Option<FinalizedStreamText>>> = None;
        let mut finalize_result: Option<Option<FinalizedStreamText>> = None;
        let stream_started = 'stream: {
            let session = match &mut engine {
                LoadedEngine::TranscribeCpp(s) => s,
                _ => break 'stream false,
            };

            // Read the backend string before beginning the stream — the
            // `Stream` borrows `session` mutably for its lifetime, so we can't
            // call `session.model()` once it exists.
            let backend = session.model().backend();

            // StreamOptions::default() uses CommitPolicy::Auto and lets the
            // family pick its own streaming strategy (no family-specific ext).
            let mut stream = match session.stream(&run_options, &StreamOptions::default()) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to begin stream: {}", e);
                    self.terminate_live_insertion_after_engine_failure();
                    break 'stream false;
                }
            };

            self.stream_active.store(true, Ordering::Release);
            self.touch_activity();
            info!(
                "Live streaming transcription started (model '{}', backend '{}')",
                model_id, backend
            );

            let mut perf = StreamPerf::new(requested_at);
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    StreamCmd::Feed(pcm) => {
                        self.touch_activity();
                        perf.record_feed(pcm.len());
                        let feed_start = Instant::now();
                        match stream.feed(&pcm) {
                            Ok(update) => {
                                perf.record_compute(feed_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                if update.committed_changed || update.tentative_changed {
                                    let text = stream.text();
                                    perf.record_emit(update.committed_changed);
                                    self.emit_stream_text(&text.committed, &text.tentative);
                                }
                                perf.maybe_log();
                            }
                            Err(e) => {
                                perf.record_compute(feed_start.elapsed());
                                warn!("stream feed failed: {}; terminating live insertion", e);
                                self.terminate_live_insertion_after_engine_failure();
                            }
                        }
                    }
                    StreamCmd::Finalize(reply) => {
                        let finalize_start = Instant::now();
                        let result = match stream.finalize() {
                            // After finalize the committed prefix holds the full
                            // text; display() = committed + tentative is the safe read.
                            Ok(update) => {
                                perf.record_compute(finalize_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                // In auto mode the model's own LID is the best
                                // remaining evidence; the snapshot is only
                                // materialized when it can change the outcome.
                                let output_language = match &output_language {
                                    OutputLanguageEvidence::Unknown => {
                                        with_model_detected_language(
                                            OutputLanguageEvidence::Unknown,
                                            stream.snapshot().language,
                                        )
                                    }
                                    resolved => resolved.clone(),
                                };
                                Some(FinalizedStreamText {
                                    text: stream.text().full,
                                    output_language,
                                    supported_languages: languages.clone(),
                                    benchmark_timing: perf
                                        .snapshot(finalize_start.elapsed(), requested_at.elapsed()),
                                })
                            }
                            Err(e) => {
                                perf.record_compute(finalize_start.elapsed());
                                error!(
                                    "stream finalize failed: {}; falling back to batch transcription",
                                    e
                                );
                                self.terminate_live_insertion_after_engine_failure();
                                None
                            }
                        };
                        let chars = match &result {
                            Some(finalized) => finalized.text.len(),
                            _ => 0,
                        };
                        perf.log_finalized(chars);
                        finalize_reply = Some(reply);
                        finalize_result = Some(result);
                        break;
                    }
                    StreamCmd::Cancel => {
                        stream.reset();
                        break;
                    }
                }
            }

            true
        };
        // `stream` + the `&mut engine` borrow are released here.

        if !stream_started {
            // Stream never began (model doesn't support streaming or begin
            // failed); drain so the finalize handshake still completes and the
            // caller falls back to batch transcription. Return the engine first
            // so the fallback can immediately use it.
            self.return_engine(engine, &model_id);
            drain_until_finalize(rx);
            return;
        }

        self.return_engine(engine, &model_id);
        if let (Some(reply), Some(result)) = (finalize_reply, finalize_result) {
            let _ = reply.send(result);
        }
        // `_worker` drops here, clearing this worker's active/lease flags after
        // the engine has been returned to the pool.
    }

    /// Mark an accelerated transcribe.cpp runtime as unhealthy for this app run.
    /// Returns true when the caller should drop the current engine rather than
    /// returning it to the pool. Saved settings are never changed.
    fn arm_runtime_cpu_fallback(&self, model_id: &str, backend: Option<&str>) -> bool {
        let mut runtime_metadata = self
            .runtime_metadata
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !mark_runtime_health_failure(&mut runtime_metadata, backend) {
            return false;
        }
        drop(runtime_metadata);
        let backend = backend.expect("runtime fallback requires an accelerated backend");

        self.force_cpu_for_run.store(true, Ordering::Release);
        let mut current_model = self
            .current_model_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if current_model.as_deref() == Some(model_id) {
            *current_model = None;
        }
        warn!(
            "Transcription runtime backend '{}' became unhealthy for model '{}'; CPU fallback is armed for the remainder of this app run and the saved accelerator preference is unchanged",
            backend, model_id
        );
        true
    }

    /// Return the leased engine to the mutex, unless the model was switched or
    /// unloaded during transcription (in which case the stale engine is dropped).
    fn return_engine(&self, engine: LoadedEngine, expected_model_id: &str) {
        let still_current =
            self.current_model_id.lock().unwrap().as_deref() == Some(expected_model_id);
        if still_current {
            *self.lock_engine() = Some(engine);
        } else {
            info!(
                "Model changed/unloaded during transcription; dropping stale engine (was '{}')",
                expected_model_id
            );
            // `engine` drops here, freeing its resources.
        }
    }

    /// Flush the active stream and return its final, post-filtered text.
    ///
    /// `Ok(None)` means no usable stream was active and the caller may fall back
    /// to batch transcription. `Err` means finalize itself failed or timed out.
    /// A timeout may still leave the worker holding the engine, so callers
    /// should surface it instead of immediately starting a batch fallback.
    pub fn finalize_stream(&self) -> Result<Option<String>> {
        Ok(self
            .finalize_stream_with_benchmark_timing()?
            .map(|(text, _)| text))
    }

    pub(crate) fn finalize_stream_with_benchmark_timing(
        &self,
    ) -> Result<Option<(String, StreamBenchmarkTiming)>> {
        let Some(tx) = self.router.take() else {
            return Ok(None);
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send(StreamCmd::Finalize(reply_tx)).is_err() {
            return Ok(None);
        }
        let finalized = match reply_rx.recv_timeout(STREAM_FINALIZE_REPLY_TIMEOUT) {
            Ok(Some(finalized)) => finalized,
            Ok(None) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.stream_active.store(false, Ordering::Release);
                return Err(anyhow::anyhow!(
                    "Timed out waiting {:?} for live transcription to finalize",
                    STREAM_FINALIZE_REPLY_TIMEOUT
                ));
            }
        };

        // Flush only the remaining raw append-safe tail before any whole-text
        // normalization/correction. If a live prefix was revised or focus moved,
        // the ledger fails closed and the action suppresses a generic re-paste.
        if let Some(attempt) = self.finalize_live_insertion_raw(&finalized.text) {
            self.execute_live_insertion_attempt(attempt);
        }

        let settings = get_settings(&self.app_handle);
        // Streaming models do not receive a decode prompt, so custom words
        // always go through the shared fuzzy post-correction path.
        let filtered = post_process_transcription_text(
            finalized.text,
            &settings,
            false,
            &finalized.output_language,
            &finalized.supported_languages,
        );

        self.maybe_unload_immediately("streaming transcription");
        Ok(Some((filtered, finalized.benchmark_timing)))
    }

    fn wait_for_stream_worker_release(&self, timeout: Duration) -> bool {
        wait_for_stream_worker_release_state(
            self.router.as_ref(),
            self.active_stream_worker.as_ref(),
            self.active_engine_lease.as_ref(),
            timeout,
        )
    }

    fn quiesce_stream_worker(&self, timeout: Duration) -> bool {
        quiesce_stream_state(
            self.router.as_ref(),
            self.active_stream_worker.as_ref(),
            self.active_engine_lease.as_ref(),
            self.stream_active.as_ref(),
            timeout,
        )
    }

    /// Feed a fixed 16 kHz WAV buffer through the same streaming route used by
    /// live capture, paced at real time. Streaming-capable engines report live
    /// milestones; engines that cannot stream cleanly fall back to one final
    /// batch decode and intentionally report no partial/cadence samples.
    pub fn benchmark_fixed_audio_with_reference(
        &self,
        audio: &[f32],
        frame_ms: u64,
        reference: Option<&str>,
    ) -> Result<TranscriptionBenchmarkSample> {
        if audio.is_empty() {
            return Err(anyhow::anyhow!("benchmark fixture contains no audio"));
        }
        let frame_ms = frame_ms.max(1);
        let frame_samples = ((16_000_u64 * frame_ms) / 1_000).max(1) as usize;
        let total_started = Instant::now();

        self.start_stream();
        for chunk in audio.chunks(frame_samples) {
            self.router.feed(chunk);
            thread::sleep(Duration::from_secs_f64(chunk.len() as f64 / 16_000.0));
        }

        self.finish_benchmark_audio(audio, total_started, reference)
    }

    /// Finish a live microphone benchmark after `start_stream` and the recorder's
    /// real-time callback have fed the active `StreamRouter` during capture.
    pub fn benchmark_live_capture(
        &self,
        audio: &[f32],
        total_started: Instant,
        reference: Option<&str>,
    ) -> Result<TranscriptionBenchmarkSample> {
        if audio.is_empty() {
            self.cancel_stream();
            return Err(anyhow::anyhow!("live benchmark captured no audio"));
        }
        self.finish_benchmark_audio(audio, total_started, reference)
    }

    fn finish_benchmark_audio(
        &self,
        audio: &[f32],
        total_started: Instant,
        reference: Option<&str>,
    ) -> Result<TranscriptionBenchmarkSample> {
        let audio_ms = ((audio.len() as u64) * 1_000) / 16_000;
        let finalize_started = Instant::now();
        if let Some((text, mut timing)) = self.finalize_stream_with_benchmark_timing()? {
            timing.finalization_tail_ms = finalize_started.elapsed().as_millis() as u64;
            timing.total_ms = total_started.elapsed().as_millis() as u64;
            let worker_released = self.wait_for_stream_worker_release(Duration::from_secs(1));
            return Ok(TranscriptionBenchmarkSample {
                mode: "streaming".to_string(),
                audio_ms,
                first_partial_ms: timing.first_partial_ms,
                committed_cadence_ms: timing.committed_cadence_ms,
                finalization_tail_ms: timing.finalization_tail_ms,
                total_ms: timing.total_ms,
                worker_released,
                word_error_rate_milli: reference
                    .map(|expected| benchmark_word_error_rate_milli(expected, &text)),
            });
        }

        let batch_started = Instant::now();
        let text = self.transcribe(audio.to_vec())?;
        let batch_ms = batch_started.elapsed().as_millis() as u64;
        let worker_released = self.wait_for_stream_worker_release(Duration::from_secs(1));
        Ok(TranscriptionBenchmarkSample {
            mode: "final_only".to_string(),
            audio_ms,
            first_partial_ms: None,
            committed_cadence_ms: Vec::new(),
            finalization_tail_ms: batch_ms,
            total_ms: total_started.elapsed().as_millis() as u64,
            worker_released,
            word_error_rate_milli: reference
                .map(|expected| benchmark_word_error_rate_milli(expected, &text)),
        })
    }

    /// Abandon any active stream without producing text (e.g. on cancel).
    pub fn cancel_stream(&self) {
        // Mark insertion cancelled before waiting so no newly observed text can
        // plan another append while the streaming worker is winding down.
        self.cancel_live_insertion();
        if !self.quiesce_stream_worker(STREAM_WORKER_QUIESCE_TIMEOUT) {
            warn!(
                "Timed out waiting {:?} for cancelled streaming worker to release",
                STREAM_WORKER_QUIESCE_TIMEOUT
            );
        }
        self.clear_live_insertion();
    }

    /// Emit a working-phase event to the streaming overlay (spinner + label).
    pub fn emit_stream_working(&self, kind: StreamWorkKind) {
        let _ = StreamPhaseEvent {
            phase: StreamPhase::Working,
            kind: Some(kind),
        }
        .emit(&self.app_handle);
    }

    fn emit_stream_text(&self, committed: &str, tentative: &str) {
        let event = StreamTextEvent {
            committed: committed.to_string(),
            tentative: tentative.to_string(),
        };
        // Tentative text is intentionally absent from the ledger API. Only the
        // append-safe committed prefix can reach the native input adapter.
        if let Some(attempt) = self.observe_live_stream_text(&event) {
            self.execute_live_insertion_attempt(attempt);
        }
        let _ = event.emit(&self.app_handle);
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();
        let audio_len = audio.len();

        debug!("Audio vector length: {}", audio_len);

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        // Validate against the model that's actually loaded (which can differ
        // from settings.selected_model when a caller loaded a specific model —
        // e.g. the --transcribe-file path's --model), not the persisted
        // selection.
        let active_model = self
            .get_current_model()
            .unwrap_or_else(|| settings.selected_model.clone());
        // Resolve the persisted language *intent* into the language this model
        // will actually use. The coercion is capability-aware (a must-pick model
        // never receives "auto") and computed fresh here — it is never written
        // back to settings, so the intent survives switching models and back.
        let validated_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &active_model);
        if validated_language != settings.selected_language {
            debug!(
                "Language intent '{}' resolved to '{}' for model '{}'",
                settings.selected_language, validated_language, active_model
            );
        }

        // Whether the loaded transcribe-cpp model advertises
        // Feature::InitialPrompt. Informational (logged below); the whisper
        // run extension and the fuzzy-correction skip are gated on
        // `model_is_whisper` instead, since non-whisper archs can advertise
        // the feature while rejecting the whisper-kind extension.
        let mut model_takes_initial_prompt = false;
        let mut vocabulary_prompted = false;
        // Whether the loaded model is actually whisper-family (arch string).
        // Non-whisper archs (e.g. Voxtral Small) can advertise
        // Feature::InitialPrompt yet reject the whisper-kind run extension
        // with INVALID_ARG, so the whisper extension must be gated on the
        // arch, not on the feature (see #1601).
        let mut model_is_whisper = false;

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let (result, output_language, model_languages) = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            // Probe live transcribe-cpp capabilities once (cheap GGUF-metadata
            // reads); the loaded session is the source of truth, not the
            // ModelManager copy. The whisper run extension is kind-tagged, so
            // non-whisper archs (parakeet, voxtral, …) reject it with
            // INVALID_ARG; attach it — and translate — only where supported.
            let mut model_supports_translate = false;
            let mut model_languages = self
                .model_manager
                .get_model_info(&active_model)
                .map(|info| info.supported_languages)
                .unwrap_or_default();
            let mut output_was_translated = false;
            let mut applied_language_hint: Option<String> = None;
            let mut model_detected_language: Option<String> = None;
            if let LoadedEngine::TranscribeCpp(session) = &engine {
                let model = session.model();
                let caps = model.capabilities();
                model_takes_initial_prompt = model.supports(Feature::InitialPrompt);
                model_is_whisper = model.arch() == "whisper";
                model_supports_translate = caps.supports_translate;
                model_languages = caps.languages;
                debug!(
                    "transcribe-cpp model '{}' on '{}': initial_prompt={}, translate={}, languages={:?}",
                    settings.selected_model,
                    model.backend(),
                    model_takes_initial_prompt,
                    model_supports_translate,
                    model_languages
                );
            }

            let transcribe_cpp_runtime_backend = match &engine {
                LoadedEngine::TranscribeCpp(session) => Some(session.model().backend().to_string()),
                _ => None,
            };

            let transcribe_result = catch_unwind(AssertUnwindSafe(|| -> Result<String> {
                match &mut engine {
                    LoadedEngine::TranscribeCpp(session) => {
                        // Only whisper-kind runs can carry the extension, and the
                        // live model must also advertise InitialPrompt. The prompt
                        // itself is bounded/sanitized and contains canonical written
                        // forms only. Unsupported engines keep vocabulary local and
                        // use deterministic post-correction below.
                        let vocabulary_prompt = if model_is_whisper && model_takes_initial_prompt {
                            build_vocabulary_prompt(
                                &settings.vocabulary_v1,
                                &settings.custom_words,
                                Some(&validated_language),
                            )
                        } else {
                            None
                        };
                        vocabulary_prompted = vocabulary_prompt.is_some();
                        let family = vocabulary_prompt.map(|initial_prompt| {
                            RunExtension::Whisper(WhisperRunOptions {
                                initial_prompt: Some(initial_prompt),
                                ..Default::default()
                            })
                        });

                        let run_plan = transcribe_cpp_run_plan(
                            settings.translate_to_english,
                            &validated_language,
                            &model_languages,
                            model_supports_translate,
                        );
                        output_was_translated = run_plan.target_language.as_deref() == Some("en");
                        applied_language_hint = run_plan.language.clone();

                        let run_options = RunOptions {
                            task: run_plan.task,
                            language: run_plan.language,
                            target_language: run_plan.target_language,
                            family,
                            ..Default::default()
                        };

                        debug!(
                            "transcribe-cpp run: task={:?}, language={:?}, initial_prompt={}",
                            run_options.task,
                            run_options.language,
                            run_options.family.is_some()
                        );

                        session
                            .run(&audio, &run_options)
                            .map(|t| {
                                // Whisper's audio-based LID (auto mode only;
                                // `None` when a language hint was passed).
                                model_detected_language = t.language;
                                t.text
                            })
                            .map_err(|e| {
                                anyhow::anyhow!("transcribe-cpp transcription failed: {}", e)
                            })
                    }
                    LoadedEngine::Parakeet(parakeet_engine) => {
                        let params = ParakeetParams {
                            timestamp_granularity: Some(TimestampGranularity::Segment),
                            ..Default::default()
                        };
                        parakeet_engine
                            .transcribe_with(&audio, &params)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Parakeet transcription failed: {}", e))
                    }
                    LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                    LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| {
                            anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                        }),
                    LoadedEngine::SenseVoice(sense_voice_engine) => {
                        let language = match normalize_cjk_language(&validated_language) {
                            "zh" => Some("zh".to_string()),
                            "en" => Some("en".to_string()),
                            "ja" => Some("ja".to_string()),
                            "ko" => Some("ko".to_string()),
                            "yue" => Some("yue".to_string()),
                            _ => None,
                        };
                        applied_language_hint = language.clone();
                        let params = SenseVoiceParams {
                            language,
                            use_itn: Some(true),
                        };
                        sense_voice_engine
                            .transcribe_with(&audio, &params)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("SenseVoice transcription failed: {}", e))
                    }
                    LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
                    LoadedEngine::Canary(canary_engine) => {
                        output_was_translated = settings.translate_to_english;
                        let lang = if validated_language == "auto" {
                            None
                        } else {
                            Some(validated_language.clone())
                        };
                        applied_language_hint = lang.clone();
                        let options = TranscribeOptions {
                            language: lang,
                            translate: settings.translate_to_english,
                            ..Default::default()
                        };
                        canary_engine
                            .transcribe(&audio, &options)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
                    }
                    LoadedEngine::Cohere(cohere_engine) => {
                        let lang = if validated_language == "auto" {
                            None
                        } else {
                            Some(normalize_cjk_language(&validated_language).to_string())
                        };
                        applied_language_hint = lang.clone();
                        let options = TranscribeOptions {
                            language: lang,
                            ..Default::default()
                        };
                        cohere_engine
                            .transcribe(&audio, &options)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Cohere transcription failed: {}", e))
                    }
                }
            }));

            let text = match transcribe_result {
                Ok(Ok(text)) => {
                    // Success: return the engine unless a model switch/unload
                    // invalidated it while it was in use.
                    self.return_engine(engine, &active_model);
                    text
                }
                Ok(Err(err)) => {
                    let accelerated_runtime_failed = self.arm_runtime_cpu_fallback(
                        &active_model,
                        transcribe_cpp_runtime_backend.as_deref(),
                    );
                    if !accelerated_runtime_failed {
                        // Ordinary model/input errors keep the engine available.
                        self.return_engine(engine, &active_model);
                    }
                    // Accelerated runtime failures deliberately drop the suspect
                    // engine. The next persisted load is forced to CPU for this
                    // process, while settings remain untouched.
                    return Err(err);
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = panic_payload_message(panic_payload.as_ref());
                    self.arm_runtime_cpu_fallback(
                        &active_model,
                        transcribe_cpp_runtime_backend.as_deref(),
                    );
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt.
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            };

            let output_language = with_model_detected_language(
                resolve_output_language_evidence(
                    &settings,
                    applied_language_hint.as_deref(),
                    &model_languages,
                    output_was_translated,
                ),
                model_detected_language,
            );
            debug!("Output language evidence: {:?}", output_language);

            (text, output_language, model_languages)
        };

        // Deterministic aliases/replacements always run. The bounded fuzzy
        // fallback runs only when vocabulary was not actually sent to a prompt-
        // capable model; streaming/unsupported engines therefore never pretend
        // to have native vocabulary support.
        let filtered_result = post_process_transcription_text(
            result,
            &settings,
            vocabulary_prompted,
            &output_language,
            &model_languages,
        );

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        // Real-time factor. Input PCM is 16 kHz mono, so audio length in seconds
        // is samples / 16000. `speedup` is audio_secs / elapsed_secs — e.g. 4.00x
        // means transcribed 4x faster than real time
        let elapsed_secs = (et - st).as_secs_f64();
        let audio_secs = audio_len as f64 / 16_000.0;
        let speedup = real_time_factor(audio_secs, elapsed_secs);
        info!(
            "Transcription completed in {:.2}s for {:.2}s of audio ({:.2}x real-time){}",
            elapsed_secs, audio_secs, speedup, translation_note
        );

        let final_result = filtered_result;

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!(
                "Transcription result: {}",
                crate::utils::redact_text(&final_result)
            );
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }
}

struct StreamPerf {
    requested_at: Instant,
    first_partial_ms: Option<u64>,
    committed_update_ms: Vec<u64>,
    feed_count: u64,
    emit_count: u64,
    streamed_samples: u64,
    stream_compute_elapsed: Duration,
    last_log: Instant,
    latest_revision: i32,
    latest_input_received_ms: i64,
    latest_audio_committed_ms: i64,
    latest_buffered_ms: i64,
}

impl StreamPerf {
    fn new(requested_at: Instant) -> Self {
        Self {
            requested_at,
            first_partial_ms: None,
            committed_update_ms: Vec::new(),
            feed_count: 0,
            emit_count: 0,
            streamed_samples: 0,
            stream_compute_elapsed: Duration::ZERO,
            last_log: Instant::now(),
            latest_revision: 0,
            latest_input_received_ms: 0,
            latest_audio_committed_ms: 0,
            latest_buffered_ms: 0,
        }
    }

    fn record_feed(&mut self, samples: usize) {
        self.feed_count += 1;
        self.streamed_samples += samples as u64;
    }

    fn record_compute(&mut self, elapsed: Duration) {
        self.stream_compute_elapsed += elapsed;
    }

    fn record_update(
        &mut self,
        revision: i32,
        input_received_ms: i64,
        audio_committed_ms: i64,
        buffered_ms: i64,
    ) {
        self.latest_revision = revision;
        self.latest_input_received_ms = input_received_ms;
        self.latest_audio_committed_ms = audio_committed_ms;
        self.latest_buffered_ms = buffered_ms;
    }

    fn record_emit(&mut self, committed_changed: bool) {
        self.emit_count += 1;
        let elapsed_ms = self.requested_at.elapsed().as_millis() as u64;
        self.first_partial_ms.get_or_insert(elapsed_ms);
        if committed_changed {
            self.committed_update_ms.push(elapsed_ms);
        }
    }

    fn snapshot(&self, finalization_tail: Duration, total: Duration) -> StreamBenchmarkTiming {
        let committed_cadence_ms = self
            .committed_update_ms
            .windows(2)
            .map(|pair| pair[1].saturating_sub(pair[0]))
            .collect();
        StreamBenchmarkTiming {
            first_partial_ms: self.first_partial_ms,
            committed_cadence_ms,
            finalization_tail_ms: finalization_tail.as_millis() as u64,
            total_ms: total.as_millis() as u64,
        }
    }

    fn maybe_log(&mut self) {
        if self.last_log.elapsed() < STREAM_PERF_LOG_INTERVAL {
            return;
        }

        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        debug!(
            "Live preview perf: {:.2}s streamed audio, {:.2}s model compute ({:.2}x real-time), \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted",
            audio_secs,
            compute_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
        );
        self.last_log = Instant::now();
    }

    fn log_finalized(&self, chars: usize) {
        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        info!(
            "Live preview finalized in {:.2}s model compute for {:.2}s streamed audio ({:.2}x real-time): \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted, {} chars",
            compute_secs,
            audio_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
            chars
        );
    }

    fn audio_secs(&self) -> f64 {
        self.streamed_samples as f64 / 16_000.0
    }

    fn compute_secs(&self) -> f64 {
        self.stream_compute_elapsed.as_secs_f64()
    }
}

fn real_time_factor(audio_secs: f64, compute_secs: f64) -> f64 {
    if compute_secs > 0.0 {
        audio_secs / compute_secs
    } else {
        0.0
    }
}

fn normalize_cjk_language(language: &str) -> &str {
    match language {
        "zh-Hans" | "zh-Hant" => "zh",
        other => other,
    }
}

/// Resolve the persisted language intent into the language a specific model can
/// use without writing the coerced value back to settings.
fn effective_language_for_model(
    settings: &AppSettings,
    model_manager: &ModelManager,
    model_id: &str,
) -> String {
    match model_manager.get_model_info(model_id) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

/// Resolve how confidently Handy knows the language of the text produced by a
/// transcription run. The UI language is deliberately not part of this
/// decision.
fn resolve_output_language_evidence(
    settings: &AppSettings,
    applied_language_hint: Option<&str>,
    supported_languages: &[String],
    translated_to_english: bool,
) -> OutputLanguageEvidence {
    if translated_to_english {
        return OutputLanguageEvidence::TranslatedToEnglish;
    }

    // Stored language intent is only evidence when this specific engine run
    // actually received the hint. Some multilingual engines (notably Parakeet
    // V3) always auto-detect and ignore Handy's selection; transcribe-cpp also
    // drops a requested hint when the loaded model does not advertise it.
    if let Some(language) = applied_language_hint.filter(|lang| !lang.is_empty() && *lang != "auto")
    {
        if settings.selected_language != "auto"
            && crate::managers::model::canonical_language_code(&settings.selected_language)
                == crate::managers::model::canonical_language_code(language)
        {
            return OutputLanguageEvidence::UserSelected(language.to_string());
        }

        // The engine may have required a concrete fallback even though the
        // user's persisted language was auto or unsupported.
        return OutputLanguageEvidence::ModelConstrained(language.to_string());
    }

    // A single-language model has a known output language without needing a
    // selectable language hint.
    if let [language] = supported_languages {
        return OutputLanguageEvidence::ModelConstrained(language.clone());
    }

    OutputLanguageEvidence::Unknown
}

/// Upgrade [`OutputLanguageEvidence::Unknown`] with the language the model
/// itself detected during the run (audio-based LID, e.g. Whisper in auto
/// mode). Stronger evidence resolved before the run is never overridden.
fn with_model_detected_language(
    evidence: OutputLanguageEvidence,
    detected: Option<String>,
) -> OutputLanguageEvidence {
    match (evidence, detected) {
        (OutputLanguageEvidence::Unknown, Some(language))
            if !language.is_empty() && language != "auto" =>
        {
            OutputLanguageEvidence::ModelDetected(language)
        }
        (evidence, _) => evidence,
    }
}

struct TranscribeCppRunPlan {
    task: Task,
    language: Option<String>,
    target_language: Option<String>,
}

/// Build the transcribe-cpp language/task options shared by batch and live
/// streaming paths.
fn transcribe_cpp_run_plan(
    translate_to_english: bool,
    effective_language: &str,
    model_languages: &[String],
    model_supports_translate: bool,
) -> TranscribeCppRunPlan {
    let requested_language = match effective_language {
        "auto" => None,
        other => Some(normalize_cjk_language(other).to_string()),
    };
    // Only pass a language the loaded model actually advertises (per
    // capabilities().languages); otherwise auto-detect rather than failing with
    // UNSUPPORTED_LANGUAGE. Language-agnostic models report an empty list, so
    // they always stay on auto.
    let language = requested_language.filter(|lang| model_languages.iter().any(|l| l == lang));
    let (task, target_language) = cpp_translation_task(
        translate_to_english,
        model_supports_translate,
        language.as_deref(),
    );

    TranscribeCppRunPlan {
        task,
        language,
        target_language,
    }
}

fn post_process_transcription_text(
    raw: String,
    settings: &AppSettings,
    custom_words_already_prompted: bool,
    output_language: &OutputLanguageEvidence,
    supported_languages: &[String],
) -> String {
    fail_open_text_transform(raw, |raw| {
        let vocabulary_result = apply_vocabulary_corrections(
            &raw,
            &settings.vocabulary_v1,
            &settings.custom_words,
            settings.word_correction_threshold,
            output_language,
            !custom_words_already_prompted,
        );
        if vocabulary_result.metadata.applied() {
            // Attribution is deliberately metadata-only: never log transcript,
            // aliases, written forms, or replacement contents.
            debug!(
                "deterministic_text_operation=vocabulary aliases={} replacements={} fuzzy={}",
                vocabulary_result.metadata.alias_replacements,
                vocabulary_result.metadata.scoped_replacements,
                vocabulary_result.metadata.fuzzy_applied
            );
        }
        let corrected = vocabulary_result.text;

        // Last-resort language evidence: confidence-gated detection from the
        // transcribed text itself, constrained to the model's languages. Only
        // consulted when it can change the outcome (built-in gated fillers).
        let output_language = match output_language {
            OutputLanguageEvidence::Unknown
                if settings.filler_word_removal_enabled
                    && settings.custom_filler_words.is_none() =>
            {
                match detect_output_language(&corrected, supported_languages) {
                    Some(language) => {
                        debug!("Text-based language detection resolved '{}'", language);
                        OutputLanguageEvidence::TextDetected(language)
                    }
                    None => OutputLanguageEvidence::Unknown,
                }
            }
            other => other.clone(),
        };

        let without_fillers = remove_filler_words(
            &corrected,
            &output_language,
            &settings.custom_filler_words,
            settings.filler_word_removal_enabled,
        );

        normalize_transcription_output(&without_fillers)
    })
}

/// Optional text cleanup must never discard a successful model result. The
/// transform is pure and owns its input, so recovering the untouched text is
/// safe even if a bug in custom-word or filler filtering unwinds.
fn fail_open_text_transform<F>(raw: String, transform: F) -> String
where
    F: FnOnce(String) -> String,
{
    let fallback = raw.clone();
    match catch_unwind(AssertUnwindSafe(|| transform(raw))) {
        Ok(processed) => processed,
        Err(payload) => {
            error!(
                "Optional transcription text post-processing panicked: {}; using the raw transcription",
                panic_payload_message(payload.as_ref())
            );
            fallback
        }
    }
}

/// Decide a transcribe-cpp run's task + translation target from settings.
///
/// "Translate to English" only fires where the model advertises translation.
/// Unlike transcribe-rs (which forces the target to English itself when its
/// `translate` flag is set), transcribe-cpp requires an explicit
/// `target_language`: a null target defaults to the *source*, so a non-English
/// source silently becomes e.g. es→es and Canary rejects the unadvertised pair.
/// An English source is skipped entirely — en→en is not a real translation, and
/// it's reachable by default since auto-detect-less models coerce intent to "en".
///
/// Returns `(task, target_language)` ready to drop into `RunOptions`.
fn cpp_translation_task(
    translate_to_english: bool,
    model_supports_translate: bool,
    source_language: Option<&str>,
) -> (Task, Option<String>) {
    let translate_to_en =
        translate_to_english && model_supports_translate && source_language != Some("en");
    if translate_to_en {
        (Task::Translate, Some("en".to_string()))
    } else {
        (Task::Transcribe, None)
    }
}

/// Drain a stream command channel, ignoring fed audio, until the caller
/// finalizes or cancels. Used when streaming can't actually run (model not
/// loaded / not streaming-capable) so the finalize handshake still completes
/// and the caller falls back to batch transcription.
fn drain_until_finalize(rx: mpsc::Receiver<StreamCmd>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Feed(_) => {}
            StreamCmd::Finalize(reply) => {
                let _ = reply.send(None);
                break;
            }
            StreamCmd::Cancel => break,
        }
    }
}

fn request_stream_cancel(router: &StreamRouter, stream_active: &AtomicBool) {
    if let Some(tx) = router.take() {
        let _ = tx.send(StreamCmd::Cancel);
    }
    stream_active.store(false, Ordering::Release);
}

fn wait_for_stream_worker_release_state(
    router: &StreamRouter,
    active_stream_worker: &AtomicU64,
    active_engine_lease: &AtomicU64,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        let released = !router.is_open()
            && active_stream_worker.load(Ordering::Acquire) == 0
            && active_engine_lease.load(Ordering::Acquire) == 0;
        if released {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(2));
    }
}

fn quiesce_stream_state(
    router: &StreamRouter,
    active_stream_worker: &AtomicU64,
    active_engine_lease: &AtomicU64,
    stream_active: &AtomicBool,
    timeout: Duration,
) -> bool {
    request_stream_cancel(router, stream_active);
    wait_for_stream_worker_release_state(router, active_stream_worker, active_engine_lease, timeout)
}

/// Initialize the transcribe-cpp native backend once at startup: route native +
/// ggml diagnostics into the `log` facade and register compute backend modules.
/// In a static build (macOS Metal) `init_backends_default` is a harmless no-op;
/// in a `dynamic-backends` build it loads the per-ISA CPU / GPU modules. Must run
/// before the first model load.
pub fn init_transcribe_backend() {
    transcribe_cpp::init_logging();
    match transcribe_cpp::init_backends_default() {
        Ok(()) => {
            if transcribe_gpu_disabled_for_host() {
                warn!(
                    "Windows x64 build is running under emulation on an ARM64 host; \
                     disabling transcribe.cpp GPU acceleration and using CPU"
                );
            }
            let devices = transcribe_compute_devices();
            info!(
                "transcribe-cpp initialized with {} compute device(s): [{}]",
                devices.len(),
                devices
                    .iter()
                    .map(|d| format!("{} ({})", d.name, d.kind))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Err(e) => warn!("Failed to initialize transcribe-cpp backends: {}", e),
    }
}

/// Human-readable list of the transcribe-cpp compute devices registered at
/// startup, for the `--list-devices` flag. The reported `index` is the
/// value to pass to `--device-index`. Backends must be initialized first
/// (see [`init_transcribe_backend`]).
pub fn describe_compute_devices() -> Vec<String> {
    transcribe_compute_devices()
        .into_iter()
        .map(|d| {
            let idx = d
                .index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".to_string());
            let name = if d.description.is_empty() {
                d.name
            } else {
                d.description
            };
            let vram_mb = d.memory_total / (1024 * 1024);
            format!(
                "index={} kind={} name={} vram={}MB",
                idx, d.kind, name, vram_mb
            )
        })
        .collect()
}

/// Resolve a `--list-devices` registry index to an exact opaque device handle
/// for a transcribe-cpp model load (the `--device-index` flag). In 0.2 index 0
/// is an exact selection too; only an omitted index requests automatic device
/// selection. Errors if the index isn't a registered, loadable primary device.
fn resolve_device_index(index: usize) -> Result<(Backend, Option<transcribe_cpp::Device>)> {
    let device = transcribe_compute_devices()
        .into_iter()
        .find(|d| d.index == Some(index))
        .ok_or_else(|| {
            anyhow::anyhow!("No compute device with index {index} (see --list-devices)")
        })?;
    if matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Accel | transcribe_cpp::DeviceType::Unknown
    ) {
        return Err(anyhow::anyhow!(
            "Device index {index} ({}) cannot host a model",
            device.kind
        ));
    }

    // 0.2's opaque handle makes every index, including zero, an exact
    // selection. Backend::Auto accepts any primary device and cannot conflict
    // with the selected device's vendor backend.
    Ok((Backend::Auto, Some(device)))
}

fn select_transcribe_backend_for_host(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> Backend {
    match effective_transcribe_accelerator(setting, gpu_disabled) {
        TranscribeAcceleratorSetting::Cpu => Backend::Cpu,
        TranscribeAcceleratorSetting::Auto | TranscribeAcceleratorSetting::Gpu => Backend::Auto,
    }
}

/// Resolve the effective transcribe.cpp backend when the runtime topology is
/// already known. If no usable GPU device registered (for example because no
/// Vulkan runtime is installed), use strict CPU for this load instead of asking
/// `Backend::Auto` to rediscover the same absence. The persisted setting remains
/// unchanged and is still recorded separately in history.
fn select_transcribe_backend_for_topology(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
    has_gpu_device: bool,
) -> Backend {
    if !has_gpu_device {
        Backend::Cpu
    } else {
        select_transcribe_backend_for_host(setting, gpu_disabled)
    }
}

/// Resolve the user's persisted GPU identity to a fresh opaque 0.2 device
/// handle. Registry indices and handles are process-local, so settings store a
/// key based on the backend's stable `device_id` (falling back to name for
/// backends such as Metal that do not report one).
fn resolve_gpu_device(
    setting: TranscribeAcceleratorSetting,
    gpu_device: Option<&str>,
) -> Option<transcribe_cpp::Device> {
    if transcribe_gpu_disabled_for_host() || setting != TranscribeAcceleratorSetting::Gpu {
        return None;
    }
    let gpu_device = gpu_device?;
    let resolved = transcribe_compute_devices().into_iter().find(|device| {
        is_transcribe_gpu_device(device) && transcribe_device_key(device) == gpu_device
    });
    if resolved.is_none() {
        warn!(
            "Stored transcribe GPU device '{}' is no longer available; using automatic device selection",
            gpu_device
        );
    }
    resolved
}

fn transcribe_device_key(device: &transcribe_cpp::Device) -> String {
    let (identity_kind, identity) = match device.device_id.as_deref() {
        Some(device_id) => ("id", device_id),
        None => ("name", device.name.as_str()),
    };
    serde_json::to_string(&(device.kind.as_str(), identity_kind, identity))
        .expect("transcribe device identity is always JSON serializable")
}

fn transcribe_device_label(device: &transcribe_cpp::Device) -> String {
    if device.description.is_empty() {
        device.name.clone()
    } else {
        device.description.clone()
    }
}

fn transcribe_accelerator_label(setting: TranscribeAcceleratorSetting) -> &'static str {
    match setting {
        TranscribeAcceleratorSetting::Auto => "auto",
        TranscribeAcceleratorSetting::Cpu => "cpu",
        TranscribeAcceleratorSetting::Gpu => "gpu",
    }
}

fn transcribe_backend_plan_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        _ => "auto",
    }
}

/// The user's persisted transcribe.cpp preference. The stable device identity is
/// kept separate from the accelerator mode so history can distinguish an exact
/// saved device from the load plan computed for the current hardware.
pub(crate) fn describe_saved_transcribe_preference(
    settings: &AppSettings,
) -> (String, Option<String>) {
    let device = (settings.transcribe_accelerator == TranscribeAcceleratorSetting::Gpu)
        .then(|| settings.transcribe_gpu_device.clone())
        .flatten();
    (
        transcribe_accelerator_label(settings.transcribe_accelerator).to_string(),
        device,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TranscribeGpuClass {
    Discrete,
    Integrated,
    Other,
}

fn transcribe_gpu_class(device: &transcribe_cpp::Device) -> TranscribeGpuClass {
    match device.device_type {
        transcribe_cpp::DeviceType::Gpu => TranscribeGpuClass::Discrete,
        transcribe_cpp::DeviceType::Igpu => TranscribeGpuClass::Integrated,
        _ => TranscribeGpuClass::Other,
    }
}

/// Return the discrete device to pin only when automatic selection would have
/// to choose between an integrated/shared-memory GPU and a discrete GPU. On
/// single-class systems we deliberately leave the device unset so
/// `Backend::Auto` retains transcribe-cpp's normal fallback behavior.
fn preferred_discrete_gpu_index(classes: &[TranscribeGpuClass]) -> Option<usize> {
    if !classes.contains(&TranscribeGpuClass::Integrated) {
        return None;
    }

    classes
        .iter()
        .position(|class| *class == TranscribeGpuClass::Discrete)
}

fn preferred_auto_transcribe_gpu_device(
    devices: Vec<transcribe_cpp::Device>,
) -> Option<transcribe_cpp::Device> {
    let classes = devices.iter().map(transcribe_gpu_class).collect::<Vec<_>>();
    let index = preferred_discrete_gpu_index(&classes)?;
    let selected = devices.into_iter().nth(index)?;
    info!(
        "Automatic transcribe.cpp device selection prefers discrete device '{}' over an integrated/shared-memory GPU",
        transcribe_device_label(&selected)
    );
    Some(selected)
}

/// Resolve the load plan Handy recommends from the saved preference and current
/// device topology. This is the single source of truth used both by model loads
/// and by history diagnostics, so the recorded recommendation cannot drift from
/// what the loader actually requested.
fn resolve_transcribe_load_plan(
    settings: &AppSettings,
) -> (Backend, Option<transcribe_cpp::Device>) {
    let accelerator = settings.transcribe_accelerator;
    let devices = transcribe_compute_devices();
    let has_gpu_device = devices.iter().any(is_transcribe_gpu_device);
    let device = match accelerator {
        TranscribeAcceleratorSetting::Auto => preferred_auto_transcribe_gpu_device(devices),
        TranscribeAcceleratorSetting::Gpu => {
            resolve_gpu_device(accelerator, settings.transcribe_gpu_device.as_deref())
        }
        TranscribeAcceleratorSetting::Cpu => None,
    };

    // Backend::Auto accepts an exact GPU device. Automatic selection only pins
    // a device when a mixed integrated + discrete topology needs disambiguation.
    // If no usable GPU registered at all (including a host with no Vulkan
    // runtime), make the load plan explicitly CPU while preserving the saved
    // preference for later runs where acceleration may become available.
    let backend = if device.is_some() {
        Backend::Auto
    } else {
        select_transcribe_backend_for_topology(
            accelerator,
            transcribe_gpu_disabled_for_host(),
            has_gpu_device,
        )
    };
    (backend, device)
}

/// A persisted accelerated load may recover on strict CPU for this process only.
/// Explicit CLI device selection stays strict: callers that hard-select a device
/// should see that selection fail rather than silently running somewhere else.
fn should_retry_transcribe_load_on_cpu(device_index: Option<usize>, backend: Backend) -> bool {
    device_index.is_none() && !matches!(backend, Backend::Cpu)
}

fn should_force_transcribe_cpu_for_run(force_cpu_for_run: bool, backend: Backend) -> bool {
    force_cpu_for_run && !matches!(backend, Backend::Cpu)
}

fn runtime_backend_needs_cpu_fallback(backend: &str) -> bool {
    !backend.eq_ignore_ascii_case("cpu")
}

fn mark_runtime_health_failure(
    runtime_metadata: &mut Option<TranscribeRuntimeMetadata>,
    backend: Option<&str>,
) -> bool {
    if !backend.is_some_and(runtime_backend_needs_cpu_fallback) {
        return false;
    }
    if let Some(runtime) = runtime_metadata.as_mut() {
        runtime.recovery_reason = Some("runtime_health_failure".to_string());
    }
    true
}

/// Apply the user's ORT accelerator preference to the transcribe-rs global.
/// Called on startup and before loading a model.
///
/// The transcribe.cpp (whisper-family) backend is no longer set here: it is
/// chosen at model-load time from [`select_transcribe_backend`], so changing the
/// accelerator only needs a model reload (see `reload_model_on_next_use`).
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    info!(
        "transcribe.cpp accelerator preference: {:?} (applied on next model load)",
        settings.transcribe_accelerator
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: String,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn transcribe_gpu_disabled_for_host() -> bool {
    crate::utils::is_windows_x64_emulated_on_arm64()
}

fn effective_transcribe_accelerator(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> TranscribeAcceleratorSetting {
    if gpu_disabled {
        TranscribeAcceleratorSetting::Cpu
    } else {
        setting
    }
}

fn is_transcribe_gpu_device(device: &transcribe_cpp::Device) -> bool {
    matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu
    )
}

fn transcribe_device_allowed(kind: &str, gpu_disabled: bool) -> bool {
    !gpu_disabled || matches!(kind, "cpu" | "accel")
}

fn transcribe_compute_devices() -> Vec<transcribe_cpp::Device> {
    let devices = transcribe_cpp::devices();
    let gpu_disabled = transcribe_gpu_disabled_for_host();
    if !gpu_disabled {
        return devices;
    }

    devices
        .into_iter()
        .filter(|device| transcribe_device_allowed(&device.kind, gpu_disabled))
        .collect()
}

fn available_transcribe_accelerators(gpu_disabled: bool) -> Vec<String> {
    if gpu_disabled {
        vec!["cpu".to_string()]
    } else {
        vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()]
    }
}

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    // GPU compute devices transcribe-cpp registered at startup. `id` is a
    // persistent identity key, never the process-local registry index. It uses
    // the backend's device_id where available and its name otherwise (Metal).
    // `total_vram_mb` is 0 when the backend does not report capacity.
    GPU_DEVICES.get_or_init(|| {
        transcribe_compute_devices()
            .into_iter()
            .filter(is_transcribe_gpu_device)
            .map(|d| GpuDeviceOption {
                id: transcribe_device_key(&d),
                name: transcribe_device_label(&d),
                total_vram_mb: (d.memory_total / (1024 * 1024)) as usize,
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub transcribe: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return the accelerators available to this process on its current host.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let transcribe_options = available_transcribe_accelerators(transcribe_gpu_disabled_for_host());

    AvailableAccelerators {
        transcribe: transcribe_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully.
        // Use match instead of unwrap to avoid panicking if the mutex is
        // poisoned — a panic inside Drop calls abort().
        let mut guard = match self.watcher_handle.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned watcher_handle mutex during TranscriptionManager drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        if let Some(handle) = guard.take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn languages(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|code| (*code).to_string()).collect()
    }

    #[test]
    fn benchmark_word_error_rate_is_deterministic_and_text_free() {
        assert_eq!(
            benchmark_word_error_rate_milli("Hello, world!", "hello world"),
            0
        );
        assert_eq!(
            benchmark_word_error_rate_milli("one two three", "one four three"),
            333
        );
        assert_eq!(
            benchmark_word_error_rate_milli("one two", "one two extra"),
            500
        );
        assert_eq!(benchmark_word_error_rate_milli("", ""), 0);
        assert_eq!(benchmark_word_error_rate_milli("", "unexpected"), 1000);
    }

    #[test]
    fn live_insertion_accessor_exposes_committed_text_only() {
        let event = StreamTextEvent {
            committed: "stable prefix".to_string(),
            tentative: " volatile suffix".to_string(),
        };

        assert_eq!(event.committed_for_live_insertion(), "stable prefix");
        assert!(!event.committed_for_live_insertion().contains("volatile"));
    }

    #[test]
    fn tentative_revisions_never_advance_the_committed_insertion_ledger() {
        let target = crate::paste_tx::TargetIdentity::for_test("target-a");
        let mut ledger = LiveInsertionLedger::with_session_id(
            InsertionMode::LiveCommittedExperimental,
            Some(target.clone()),
            7,
        );
        assert!(ledger.observe_speech_evidence(Some(&target)).is_none());

        let first = StreamTextEvent {
            committed: "stable prefix".to_string(),
            tentative: " first guess".to_string(),
        };
        let attempt = ledger
            .observe_committed(first.committed_for_live_insertion(), Some(&target))
            .expect("committed prefix should plan one insertion");
        assert_eq!(attempt.text, "stable prefix");
        assert!(ledger.record_attempt_result(
            attempt.session_id,
            attempt.sequence,
            LiveInsertionOutcome::Inserted,
        ));

        let revised_tentative = StreamTextEvent {
            committed: "stable prefix".to_string(),
            tentative: " completely different guess".to_string(),
        };
        assert!(ledger
            .observe_committed(
                revised_tentative.committed_for_live_insertion(),
                Some(&target),
            )
            .is_none());
        assert_eq!(ledger.inserted_committed(), "stable prefix");
        assert_eq!(ledger.attempts().len(), 1);
    }

    #[test]
    fn terminal_live_state_survives_active_ledger_clear_for_final_paste_guard() {
        let blocked_after_clear = AtomicBool::new(false);
        let mut active = Some(LiveInsertionLedger::new(
            InsertionMode::LiveCommittedExperimental,
            None,
        ));
        active.as_mut().unwrap().cancel();

        clear_live_insertion_state(&mut active, &blocked_after_clear);

        assert!(active.is_none());
        assert!(blocked_after_clear.load(Ordering::Acquire));
        assert!(live_insertion_state_blocks_final_paste(
            active.as_ref(),
            blocked_after_clear.load(Ordering::Acquire)
        ));
    }

    #[test]
    fn engine_failure_terminates_active_live_session_and_blocks_repaste() {
        let blocked_after_clear = AtomicBool::new(false);
        let mut active = Some(LiveInsertionLedger::new(
            InsertionMode::LiveCommittedExperimental,
            None,
        ));

        terminate_live_insertion_state(&mut active, &blocked_after_clear);

        assert!(active.is_none());
        assert!(blocked_after_clear.load(Ordering::Acquire));
        assert!(live_insertion_state_blocks_final_paste(
            active.as_ref(),
            blocked_after_clear.load(Ordering::Acquire)
        ));
    }

    #[test]
    fn terminal_focus_loss_clears_active_live_session_and_keeps_repaste_blocked() {
        let blocked_after_clear = AtomicBool::new(false);
        let mut ledger = LiveInsertionLedger::new(InsertionMode::LiveCommittedExperimental, None);
        ledger.observe_speech_evidence(None);
        let mut active = Some(ledger);

        clear_terminal_live_insertion_state(&mut active, &blocked_after_clear);

        assert!(active.is_none());
        assert!(blocked_after_clear.load(Ordering::Acquire));
        assert!(live_insertion_state_blocks_final_paste(
            active.as_ref(),
            blocked_after_clear.load(Ordering::Acquire)
        ));
    }

    #[test]
    fn stream_text_event_serializes_committed_and_tentative_as_distinct_fields() {
        let event = StreamTextEvent {
            committed: "stable prefix".to_string(),
            tentative: " volatile suffix".to_string(),
        };
        let value = serde_json::to_value(event).unwrap();

        assert_eq!(value["committed"], "stable prefix");
        assert_eq!(value["tentative"], " volatile suffix");
        assert_ne!(value["committed"], value["tentative"]);
    }

    #[test]
    fn initial_lazy_load_is_not_a_model_switch_but_changed_loaded_model_is() {
        assert!(!is_model_switch(None, "moonshine-base"));
        assert!(!is_model_switch(Some("moonshine-base"), "moonshine-base"));
        assert!(is_model_switch(Some("moonshine-base"), "parakeet-tdt"));
    }

    #[test]
    fn stream_worker_guard_releases_cancelled_worker_state() {
        let active_stream_worker = Arc::new(AtomicU64::new(7));
        let active_engine_lease = Arc::new(AtomicU64::new(7));
        let stream_active = Arc::new(AtomicBool::new(true));
        {
            let _guard = StreamWorkerGuard {
                worker_id: 7,
                active_stream_worker: Arc::clone(&active_stream_worker),
                active_engine_lease: Arc::clone(&active_engine_lease),
                stream_active: Arc::clone(&stream_active),
            };
        }

        assert_eq!(active_stream_worker.load(Ordering::Acquire), 0);
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 0);
        assert!(!stream_active.load(Ordering::Acquire));
    }

    #[test]
    fn stale_stream_worker_cannot_clear_new_model_switch_worker() {
        let active_stream_worker = Arc::new(AtomicU64::new(2));
        let active_engine_lease = Arc::new(AtomicU64::new(2));
        let stream_active = Arc::new(AtomicBool::new(true));
        {
            let _stale_guard = StreamWorkerGuard {
                worker_id: 1,
                active_stream_worker: Arc::clone(&active_stream_worker),
                active_engine_lease: Arc::clone(&active_engine_lease),
                stream_active: Arc::clone(&stream_active),
            };
        }

        assert_eq!(active_stream_worker.load(Ordering::Acquire), 2);
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 2);
        assert!(stream_active.load(Ordering::Acquire));

        {
            let _current_guard = StreamWorkerGuard {
                worker_id: 2,
                active_stream_worker: Arc::clone(&active_stream_worker),
                active_engine_lease: Arc::clone(&active_engine_lease),
                stream_active: Arc::clone(&stream_active),
            };
        }
        assert_eq!(active_stream_worker.load(Ordering::Acquire), 0);
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 0);
        assert!(!stream_active.load(Ordering::Acquire));
    }

    fn spawn_test_stream_worker(
        router: &StreamRouter,
        worker_id: u64,
        active_stream_worker: Arc<AtomicU64>,
        active_engine_lease: Arc<AtomicU64>,
        stream_active: Arc<AtomicBool>,
    ) -> thread::JoinHandle<()> {
        let rx = router.open();
        active_stream_worker.store(worker_id, Ordering::Release);
        active_engine_lease.store(worker_id, Ordering::Release);
        stream_active.store(true, Ordering::Release);
        thread::spawn(move || {
            let _guard = StreamWorkerGuard {
                worker_id,
                active_stream_worker,
                active_engine_lease,
                stream_active,
            };
            assert!(matches!(rx.recv(), Ok(StreamCmd::Cancel)));
        })
    }

    #[test]
    fn cancellation_quiesces_worker_and_allows_next_session_reservation() {
        let router = StreamRouter::new();
        let active_stream_worker = Arc::new(AtomicU64::new(0));
        let active_engine_lease = Arc::new(AtomicU64::new(0));
        let stream_active = Arc::new(AtomicBool::new(false));
        let worker = spawn_test_stream_worker(
            &router,
            41,
            Arc::clone(&active_stream_worker),
            Arc::clone(&active_engine_lease),
            Arc::clone(&stream_active),
        );

        assert!(quiesce_stream_state(
            &router,
            active_stream_worker.as_ref(),
            active_engine_lease.as_ref(),
            stream_active.as_ref(),
            Duration::from_secs(1),
        ));
        worker.join().unwrap();
        assert!(!router.is_open());
        assert!(!stream_active.load(Ordering::Acquire));
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 0);
        assert!(active_stream_worker
            .compare_exchange(0, 42, Ordering::AcqRel, Ordering::Acquire)
            .is_ok());
    }

    #[test]
    fn model_switch_quiesce_releases_old_lease_before_new_worker_generation() {
        let router = StreamRouter::new();
        let active_stream_worker = Arc::new(AtomicU64::new(0));
        let active_engine_lease = Arc::new(AtomicU64::new(0));
        let stream_active = Arc::new(AtomicBool::new(false));
        let worker = spawn_test_stream_worker(
            &router,
            7,
            Arc::clone(&active_stream_worker),
            Arc::clone(&active_engine_lease),
            Arc::clone(&stream_active),
        );

        assert!(quiesce_stream_state(
            &router,
            active_stream_worker.as_ref(),
            active_engine_lease.as_ref(),
            stream_active.as_ref(),
            Duration::from_secs(1),
        ));
        worker.join().unwrap();
        assert_eq!(active_stream_worker.load(Ordering::Acquire), 0);
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 0);

        let new_worker_id = 8;
        assert!(active_stream_worker
            .compare_exchange(0, new_worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok());
        active_engine_lease.store(new_worker_id, Ordering::Release);
        stream_active.store(true, Ordering::Release);
        {
            let _new_guard = StreamWorkerGuard {
                worker_id: new_worker_id,
                active_stream_worker: Arc::clone(&active_stream_worker),
                active_engine_lease: Arc::clone(&active_engine_lease),
                stream_active: Arc::clone(&stream_active),
            };
        }
        assert_eq!(active_stream_worker.load(Ordering::Acquire), 0);
        assert_eq!(active_engine_lease.load(Ordering::Acquire), 0);
        assert!(!stream_active.load(Ordering::Acquire));
    }

    #[test]
    fn stream_perf_records_timing_only_and_committed_cadence() {
        let started = Instant::now();
        let mut perf = StreamPerf::new(started);
        perf.record_emit(false);
        std::thread::sleep(Duration::from_millis(1));
        perf.record_emit(true);
        std::thread::sleep(Duration::from_millis(1));
        perf.record_emit(true);
        let timing = perf.snapshot(Duration::from_millis(4), started.elapsed());

        assert!(timing.first_partial_ms.is_some());
        assert_eq!(timing.committed_cadence_ms.len(), 1);
        assert_eq!(timing.finalization_tail_ms, 4);
        let serialized = serde_json::to_string(&timing).unwrap();
        assert!(!serialized.contains("transcript"));
        assert!(!serialized.contains("audio"));
        assert!(!serialized.contains("clipboard"));
        assert!(!serialized.contains("window_title"));
        assert!(!serialized.contains("window title"));
    }

    #[test]
    fn normal_hosts_preserve_every_transcribe_accelerator_setting() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(effective_transcribe_accelerator(setting, false), setting);
        }
        assert_eq!(
            available_transcribe_accelerators(false),
            ["auto", "cpu", "gpu"]
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Auto, false),
            Backend::Auto
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Cpu, false),
            Backend::Cpu
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Gpu, false),
            Backend::Auto
        );
        for kind in ["cpu", "accel", "metal", "cuda", "vulkan", "gpu"] {
            assert!(transcribe_device_allowed(kind, false));
        }
    }

    #[test]
    fn automatic_gpu_selection_pins_discrete_only_for_mixed_gpu_topologies() {
        use TranscribeGpuClass::{Discrete, Integrated, Other};

        assert_eq!(
            preferred_discrete_gpu_index(&[Integrated, Discrete]),
            Some(1)
        );
        assert_eq!(
            preferred_discrete_gpu_index(&[Discrete, Integrated]),
            Some(0)
        );
        assert_eq!(
            preferred_discrete_gpu_index(&[Other, Integrated, Discrete]),
            Some(2)
        );
        assert_eq!(preferred_discrete_gpu_index(&[Integrated]), None);
        assert_eq!(preferred_discrete_gpu_index(&[Discrete]), None);
        assert_eq!(preferred_discrete_gpu_index(&[Other, Discrete]), None);
    }

    #[test]
    fn no_gpu_topology_uses_cpu_without_changing_accelerator_intent() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(
                select_transcribe_backend_for_topology(setting, false, false),
                Backend::Cpu
            );
        }
        assert_eq!(
            select_transcribe_backend_for_topology(TranscribeAcceleratorSetting::Auto, false, true,),
            Backend::Auto
        );
        assert_eq!(
            select_transcribe_backend_for_topology(TranscribeAcceleratorSetting::Gpu, false, true,),
            Backend::Auto
        );
    }

    #[test]
    fn no_vulkan_startup_stays_cpu_while_saved_gpu_intent_survives() {
        let settings = AppSettings {
            transcribe_accelerator: TranscribeAcceleratorSetting::Gpu,
            transcribe_gpu_device: Some("vulkan-device-that-is-currently-absent".to_string()),
            ..Default::default()
        };
        let saved = describe_saved_transcribe_preference(&settings);

        // No registered GPU is the deterministic stand-in for a machine with no
        // Vulkan runtime/device available at startup.
        let recommended =
            select_transcribe_backend_for_topology(settings.transcribe_accelerator, false, false);
        assert_eq!(recommended, Backend::Cpu);
        assert!(!should_retry_transcribe_load_on_cpu(None, recommended));
        let runtime = transcribe_runtime_metadata("cpu".to_string(), None, None);
        assert_eq!(runtime.backend, "cpu");
        assert!(runtime.device.is_none());
        assert!(runtime.recovery_reason.is_none());
        assert_eq!(describe_saved_transcribe_preference(&settings), saved);
        assert_eq!(saved.0, "gpu");
        assert_eq!(
            saved.1.as_deref(),
            Some("vulkan-device-that-is-currently-absent")
        );
    }

    #[test]
    fn persisted_accelerated_loads_can_fallback_to_cpu_but_explicit_device_loads_stay_strict() {
        assert!(should_retry_transcribe_load_on_cpu(None, Backend::Auto));
        assert!(!should_retry_transcribe_load_on_cpu(None, Backend::Cpu));
        assert!(!should_retry_transcribe_load_on_cpu(Some(0), Backend::Auto));
    }

    #[test]
    fn startup_gpu_failure_falls_back_for_run_without_rewriting_saved_preference() {
        let settings = AppSettings {
            transcribe_accelerator: TranscribeAcceleratorSetting::Gpu,
            transcribe_gpu_device: Some("stable-discrete-gpu".to_string()),
            ..Default::default()
        };

        let saved_before = describe_saved_transcribe_preference(&settings);
        let requested_backend = Backend::Auto;
        assert!(should_retry_transcribe_load_on_cpu(None, requested_backend));

        // The retry is a load-time CPU decision, not a settings migration.
        let fallback_backend = Backend::Cpu;
        assert_eq!(fallback_backend, Backend::Cpu);
        let runtime = transcribe_runtime_metadata(
            "cpu".to_string(),
            None,
            Some("startup_gpu_fallback".to_string()),
        );
        assert_eq!(runtime.backend, "cpu");
        assert!(runtime.device.is_none());
        assert_eq!(
            runtime.recovery_reason.as_deref(),
            Some("startup_gpu_fallback")
        );
        assert_eq!(
            describe_saved_transcribe_preference(&settings),
            saved_before
        );
        assert_eq!(saved_before.0, "gpu");
        assert_eq!(saved_before.1.as_deref(), Some("stable-discrete-gpu"));
    }

    #[test]
    fn runtime_gpu_failure_preserves_saved_and_recommended_diagnostics_for_cpu_reload() {
        let settings = AppSettings {
            transcribe_accelerator: TranscribeAcceleratorSetting::Gpu,
            transcribe_gpu_device: Some("stable-discrete-gpu".to_string()),
            ..Default::default()
        };

        let saved = describe_saved_transcribe_preference(&settings);
        let recommended_backend = Backend::Auto;
        let selection_plan = TranscribeSelectionPlanMetadata {
            saved_accelerator: Some(saved.0.clone()),
            saved_gpu_device: saved.1.clone(),
            recommended_backend: transcribe_backend_plan_label(recommended_backend).to_string(),
            recommended_device: Some("Discrete GPU".to_string()),
        };

        assert!(runtime_backend_needs_cpu_fallback("vulkan"));
        assert!(should_force_transcribe_cpu_for_run(
            true,
            recommended_backend
        ));
        let actual_reload_backend = Backend::Cpu;

        // Diagnostics retain three different facts: persisted intent, Handy's
        // hardware recommendation, and the runtime backend actually used after
        // the health latch fired.
        assert_eq!(selection_plan.saved_accelerator.as_deref(), Some("gpu"));
        assert_eq!(
            selection_plan.saved_gpu_device.as_deref(),
            Some("stable-discrete-gpu")
        );
        assert_eq!(selection_plan.recommended_backend, "auto");
        assert_eq!(
            selection_plan.recommended_device.as_deref(),
            Some("Discrete GPU")
        );
        assert_eq!(transcribe_backend_plan_label(actual_reload_backend), "cpu");
        assert_eq!(describe_saved_transcribe_preference(&settings), saved);
    }

    #[test]
    fn runtime_health_latch_forces_only_accelerated_persisted_loads_to_cpu() {
        assert!(should_force_transcribe_cpu_for_run(true, Backend::Auto));
        assert!(!should_force_transcribe_cpu_for_run(true, Backend::Cpu));
        assert!(!should_force_transcribe_cpu_for_run(false, Backend::Auto));
    }

    #[test]
    fn runtime_health_downgrade_preserves_actual_runtime_and_saved_preference() {
        let settings = AppSettings {
            transcribe_accelerator: TranscribeAcceleratorSetting::Gpu,
            transcribe_gpu_device: Some("stable-discrete-gpu".to_string()),
            ..Default::default()
        };
        let saved = describe_saved_transcribe_preference(&settings);
        let mut runtime = Some(transcribe_runtime_metadata(
            "vulkan".to_string(),
            Some("Discrete GPU".to_string()),
            None,
        ));

        assert!(mark_runtime_health_failure(&mut runtime, Some("vulkan")));
        let runtime = runtime.expect("runtime snapshot survives engine drop");
        assert_eq!(runtime.backend, "vulkan");
        assert_eq!(runtime.device.as_deref(), Some("Discrete GPU"));
        assert_eq!(
            runtime.recovery_reason.as_deref(),
            Some("runtime_health_failure")
        );
        assert_eq!(describe_saved_transcribe_preference(&settings), saved);
    }

    #[test]
    fn runtime_health_downgrade_ignores_cpu_failures() {
        assert!(runtime_backend_needs_cpu_fallback("vulkan"));
        assert!(runtime_backend_needs_cpu_fallback("cuda"));
        assert!(runtime_backend_needs_cpu_fallback("metal"));
        assert!(!runtime_backend_needs_cpu_fallback("cpu"));
        assert!(!runtime_backend_needs_cpu_fallback("CPU"));
        let mut runtime = Some(transcribe_runtime_metadata(
            "cpu".to_string(),
            Some("cpu".to_string()),
            None,
        ));
        assert!(!mark_runtime_health_failure(&mut runtime, Some("cpu")));
        assert!(runtime.unwrap().recovery_reason.is_none());
    }

    #[test]
    fn saved_transcribe_preference_keeps_exact_device_identity_separate() {
        let mut settings = AppSettings {
            transcribe_accelerator: TranscribeAcceleratorSetting::Gpu,
            transcribe_gpu_device: Some("stable-device-id".to_string()),
            ..Default::default()
        };

        assert_eq!(
            describe_saved_transcribe_preference(&settings),
            ("gpu".to_string(), Some("stable-device-id".to_string()))
        );

        settings.transcribe_accelerator = TranscribeAcceleratorSetting::Auto;
        assert_eq!(
            describe_saved_transcribe_preference(&settings),
            ("auto".to_string(), None)
        );
    }

    #[test]
    fn emulated_x64_on_arm64_forces_every_transcribe_setting_to_cpu() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(
                effective_transcribe_accelerator(setting, true),
                TranscribeAcceleratorSetting::Cpu
            );
            assert_eq!(
                select_transcribe_backend_for_host(setting, true),
                Backend::Cpu
            );
        }
        assert_eq!(available_transcribe_accelerators(true), ["cpu"]);
        assert!(transcribe_device_allowed("cpu", true));
        assert!(transcribe_device_allowed("accel", true));
        for kind in ["metal", "cuda", "vulkan", "gpu", "unknown"] {
            assert!(!transcribe_device_allowed(kind, true));
        }
    }

    #[test]
    fn optional_text_transform_falls_back_to_raw_text_after_panic() {
        let raw = "原始轉錄。".to_string();
        let result = fail_open_text_transform(raw.clone(), |_| {
            panic!("simulated optional cleanup failure")
        });

        assert_eq!(result, raw);
    }

    #[test]
    fn portuguese_transcription_does_not_use_english_ui_filler_words() {
        let settings = AppSettings {
            app_language: "en".to_string(),
            selected_language: "pt-BR".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "pt"]);
        let evidence = resolve_output_language_evidence(&settings, Some("pt"), &supported, false);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn norwegian_alias_is_recorded_as_user_selected_evidence() {
        let settings = AppSettings {
            selected_language: "no".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, Some("nb"), &languages(&["nb"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("nb".to_string())
        );
    }

    #[test]
    fn auto_language_without_detection_skips_gated_filler_removal() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };
        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en", "pt"]), false);

        // Too short for a reliable text detection, so the gated "um" must
        // survive; the universal "uhm" is removed regardless.
        let result = post_process_transcription_text(
            "um uhm ok".to_string(),
            &settings,
            false,
            &evidence,
            &languages(&["en", "pt"]),
        );

        assert_eq!(evidence, OutputLanguageEvidence::Unknown);
        assert_eq!(result, "um ok");
    }

    #[test]
    fn unknown_evidence_with_confident_text_detection_removes_gated_fillers() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "um so the weather forecast said it would probably rain throughout the whole weekend"
                .to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "so the weather forecast said it would probably rain throughout the whole weekend"
        );
    }

    #[test]
    fn unknown_evidence_with_portuguese_text_preserves_um() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "eu vi um carro na rua ontem de manhã quando fui ao mercado".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "eu vi um carro na rua ontem de manhã quando fui ao mercado"
        );
    }

    #[test]
    fn model_detected_language_upgrades_unknown_evidence_only() {
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("en".to_string())),
            OutputLanguageEvidence::ModelDetected("en".to_string())
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("auto".to_string())),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, None),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(
                OutputLanguageEvidence::UserSelected("pt".to_string()),
                Some("en".to_string())
            ),
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
    }

    #[test]
    fn auto_language_uses_single_language_model_as_evidence() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn unsupported_explicit_language_uses_model_fallback_as_evidence() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("en"),
            &languages(&["en", "de"]),
            false,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn ignored_user_language_is_not_output_evidence() {
        let settings = AppSettings {
            // Parakeet V3 ignores language hints and auto-detects even when a
            // selection from the previously active model remains persisted.
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "de", "pt"]);

        let evidence = resolve_output_language_evidence(&settings, None, &supported, false);
        assert_eq!(evidence, OutputLanguageEvidence::Unknown);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn unapplied_transcribe_cpp_language_is_not_output_evidence() {
        let settings = AppSettings {
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&[]);
        let plan = transcribe_cpp_run_plan(false, "en", &supported, false);

        assert_eq!(plan.language, None);
        assert_eq!(
            resolve_output_language_evidence(
                &settings,
                plan.language.as_deref(),
                &supported,
                false,
            ),
            OutputLanguageEvidence::Unknown
        );
    }

    #[test]
    fn translated_output_is_treated_as_english() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("pt"),
            &languages(&["en", "pt"]),
            true,
        );

        assert_eq!(evidence, OutputLanguageEvidence::TranslatedToEnglish);
    }

    #[test]
    fn transcribe_cpp_run_plan_maps_chinese_variants() {
        let plan = transcribe_cpp_run_plan(false, "zh-Hant", &languages(&["zh"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("zh"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_skips_english_translation() {
        let plan = transcribe_cpp_run_plan(true, "en", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("en"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_translates_supported_non_english() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Translate));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language.as_deref(), Some("en"));
    }

    #[test]
    fn transcribe_cpp_run_plan_requires_model_translation_support() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), false);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language, None);
    }
}
