#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error, VadPolicy};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::media::RecordingMediaController;
use crate::managers::model::{EngineType, ModelManager};
use crate::managers::performance::{PerformanceManager, PerformanceSessionMetadata};
use crate::managers::transcription::StreamWorkKind;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{
    get_settings, AppSettings, InsertionMode, OverlayStyle, APPLE_INTELLIGENCE_PROVIDER_ID,
    LOCAL_CLEANUP_PROVIDER_ID,
};
use crate::shortcut;
use crate::tray::{set_tray_state, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;
use tauri::{AppHandle, Emitter};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        // Every terminal pipeline path (success, cancellation, transcription
        // error, output-handler error, or panic) ends the insertion session.
        if let Some(tm) = self.0.try_state::<Arc<TranscriptionManager>>() {
            tm.clear_live_insertion();
            tm.maybe_unload_immediately("transcription pipeline completion");
        }
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
        // The pipeline just freed its large transient buffers (captured PCM,
        // WAV copy, engine scratch); hand the cached pages back to the OS so
        // they don't sit in malloc arenas until they get swapped out (#1792).
        crate::memory::trim_freed_memory();
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Non-negotiable cleanup response contract. User prompts may customize how the
/// transcript is cleaned, but they must not relax the shape or safety of the
/// returned text.
const CLEANUP_OUTPUT_CONTRACT: &str = "Return only the cleaned transcription text. Do not add explanations, surrounding quotes, markdown/code fences, JSON/XML wrappers, or invented content.";
/// A second hard bound after the HTTP response cap. This constrains the actual
/// text field even when a server pads the JSON envelope or ignores max_tokens.
const CLEANUP_MAX_OUTPUT_CHARS: usize = 16 * 1024;

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Strip a leading `<think>...</think>` block. Some endpoints can't disable
/// reasoning, and some local servers put the reasoning text into `content`
/// instead of a separate field — without this the user would get the model's
/// chain of thought pasted along with the cleaned transcription.
fn strip_think_block(s: &str) -> &str {
    if let Some(rest) = s.trim_start().strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            return rest[end + "</think>".len()..].trim_start();
        }
    }
    s
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message,
/// then appends the fork's strict output contract so custom prompts cannot silently
/// opt out of the text-only response shape.
fn build_system_prompt(prompt_template: &str) -> String {
    let prompt = prompt_template.replace("${output}", "").trim().to_string();
    if prompt.is_empty() {
        CLEANUP_OUTPUT_CONTRACT.to_string()
    } else {
        format!("{prompt}\n\n{CLEANUP_OUTPUT_CONTRACT}")
    }
}

fn build_legacy_prompt(prompt_template: &str, transcription: &str) -> String {
    let prompt = prompt_template.replace("${output}", transcription);
    format!("{}\n\n{}", prompt.trim(), CLEANUP_OUTPUT_CONTRACT)
}

/// Normalize a successful cleanup response while enforcing the text-only contract.
/// Obvious wrappers are treated as malformed and therefore fail open to the raw
/// transcript at the caller instead of being pasted into the target application.
fn normalize_cleanup_output(output: &str) -> Option<String> {
    let output = strip_invisible_chars(strip_think_block(output));
    let output = output.trim();
    if output.is_empty() || output.chars().count() > CLEANUP_MAX_OUTPUT_CHARS {
        return None;
    }

    let fenced = output.starts_with("```") && output.ends_with("```");
    let quoted = (output.starts_with('"') && output.ends_with('"'))
        || (output.starts_with('\'') && output.ends_with('\''))
        || (output.starts_with('`') && output.ends_with('`'));
    if fenced || quoted || looks_like_cleanup_wrapper(output) {
        return None;
    }

    Some(output.to_string())
}

fn looks_like_cleanup_wrapper(output: &str) -> bool {
    // Legacy endpoints sometimes ignore the text-only instruction and return a
    // JSON object/array. Reject the wrapper rather than opportunistically mining
    // a field out of it; structured providers have their own exact schema path.
    if serde_json::from_str::<serde_json::Value>(output)
        .ok()
        .is_some_and(|value| value.is_object() || value.is_array())
    {
        return true;
    }

    let lower = output.to_ascii_lowercase();
    for tag in [
        "transcription",
        "output",
        "response",
        "result",
        "cleaned_text",
    ] {
        if lower.starts_with(&format!("<{tag}>")) && lower.ends_with(&format!("</{tag}>")) {
            return true;
        }
    }

    [
        "cleaned transcription:",
        "cleaned text:",
        "here is the cleaned transcription:",
        "here's the cleaned transcription:",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn parse_structured_cleanup_output(content: &str) -> Option<String> {
    let content = strip_think_block(content).trim();
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }

    normalize_cleanup_output(object.get(TRANSCRIPTION_FIELD)?.as_str()?)
}

/// Returns `true` when a transcription has no meaningful content to
/// post-process (empty or whitespace-only). Used to skip the post-processing
/// LLM call when nothing was actually transcribed, which would otherwise make
/// the model reply with an error message such as "you need to provide the
/// transcription".
fn is_blank_transcription(transcription: &str) -> bool {
    transcription.trim().is_empty()
}

async fn complete_unless_cancelled<F, C>(operation: F, is_cancelled: C) -> Option<F::Output>
where
    F: Future,
    C: Fn() -> bool,
{
    tokio::pin!(operation);

    loop {
        if is_cancelled() {
            return None;
        }

        if let Ok(result) =
            tokio::time::timeout(CANCELLATION_POLL_INTERVAL, operation.as_mut()).await
        {
            return Some(result);
        }
    }
}

fn should_use_streaming_overlay(style: OverlayStyle, is_streaming: bool) -> bool {
    style == OverlayStyle::Live && is_streaming
}

/// Resolve the persisted insertion preference into the safe mode for this
/// recording. Non-streaming models retain final-at-stop behavior. A request for
/// whole-transcript AI cleanup cannot coexist with committed live insertion, so
/// that session is downgraded to preview-only while cleanup remains available at
/// stop.
fn resolve_session_insertion_mode(
    configured: InsertionMode,
    experimental_enabled: bool,
    post_process: bool,
    model_supports_streaming: bool,
    positive_speech_evidence_available: bool,
) -> InsertionMode {
    if !experimental_enabled || !model_supports_streaming {
        return InsertionMode::AtStop;
    }
    if configured == InsertionMode::LiveCommittedExperimental
        && (post_process || !positive_speech_evidence_available)
    {
        return InsertionMode::PreviewOnly;
    }
    configured
}

fn should_paste_final_output(
    final_text_is_empty: bool,
    session_insertion_mode: InsertionMode,
    live_insertion_blocks_final_paste: bool,
) -> bool {
    !(final_text_is_empty
        || (session_insertion_mode == InsertionMode::LiveCommittedExperimental
            && live_insertion_blocks_final_paste))
}

fn insertion_mode_history_value(mode: InsertionMode) -> &'static str {
    match mode {
        InsertionMode::AtStop => "at_stop",
        InsertionMode::PreviewOnly => "preview_only",
        InsertionMode::LiveCommittedExperimental => "live_committed_experimental",
    }
}

fn resolve_stream_or_batch<E, F>(
    stream_result: Result<Option<String>, E>,
    batch_transcribe: F,
) -> Result<String, E>
where
    F: FnOnce() -> Result<String, E>,
{
    match stream_result {
        Ok(Some(text)) if !text.trim().is_empty() => Ok(text),
        Ok(_) => batch_transcribe(),
        Err(err) => Err(err),
    }
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> Option<String> {
    if is_blank_transcription(transcription) {
        debug!("Post-processing skipped because the transcription is empty");
        return None;
    }

    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Ask these providers to skip reasoning/thinking — post-processing rarely
    // benefits from it and it adds seconds of latency. llm_client picks the
    // field the endpoint understands and retries without it if rejected.
    let disable_reasoning = matches!(
        provider.id.as_str(),
        "custom" | "openrouter" | LOCAL_CLEANUP_PROVIDER_ID
    );

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(&prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => match normalize_cleanup_output(&result) {
                        Some(result) => {
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                        None => {
                            error!("Apple Intelligence returned malformed cleanup output; falling back to the raw transcription");
                            None
                        }
                    },
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            disable_reasoning,
        )
        .await
        {
            Ok(Some(content)) => {
                if let Some(result) = parse_structured_cleanup_output(&content) {
                    debug!(
                        "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                        provider.id,
                        result.len()
                    );
                    return Some(result);
                }

                error!(
                    "Structured output response for provider '{}' violated the cleanup contract; falling back to the raw transcription",
                    provider.id
                );
                return None;
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: replace ${output} with the actual text and append the same
    // strict response contract used by structured providers.
    let processed_prompt = build_legacy_prompt(&prompt, transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        disable_reasoning,
    )
    .await
    {
        Ok(Some(content)) => match normalize_cleanup_output(&content) {
            Some(content) => {
                debug!(
                    "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                    provider.id,
                    content.len()
                );
                Some(content)
            }
            None => {
                error!(
                    "LLM post-processing output for provider '{}' violated the cleanup contract; falling back to the raw transcription",
                    provider.id
                );
                None
            }
        },
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

async fn maybe_convert_chinese_variant(
    effective_language: &str,
    transcription: &str,
) -> Option<String> {
    // Gate on the language the model actually transcribed in (the effective
    // language), not the persisted intent. A leftover zh-Hans/zh-Hant intent
    // from a previously selected model must not run OpenCC S2T/T2S over output a
    // non-Chinese model produced — that would silently rewrite any shared CJK
    // characters (e.g. Japanese kanji) in the result.
    let is_simplified = effective_language == "zh-Hans";
    let is_traditional = effective_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("effective language is not Simplified or Traditional Chinese; skipping conversion");
        return None;
    }

    debug!(
        "Starting Chinese variant conversion using OpenCC for language: {}",
        effective_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    /// Stable selected prompt identifier used for cleanup. Unlike
    /// `post_process_prompt`, this is safe metadata and remains meaningful when
    /// the prompt body is later edited.
    pub cleanup_prompt_id: Option<String>,
    /// Cleanup-model identifier, separate from the ASR `model_id` persisted in
    /// history. This never contains transcript or application context.
    pub cleanup_model_id: Option<String>,
    pub cleanup_mode: String,
}

/// Apply cleanup as an explicitly fail-open transform. Every cleanup runtime
/// failure is represented as `None`; this helper makes the preservation rule
/// testable without a Tauri AppHandle or a live model server.
fn apply_cleanup_fail_open(
    raw_text: &str,
    cleaned_text: Option<String>,
) -> (String, Option<String>) {
    match cleaned_text {
        Some(cleaned) => (cleaned.clone(), Some(cleaned)),
        None => (raw_text.to_string(), None),
    }
}

/// Describe the cleanup path requested for a history row without persisting
/// transcript content or provider secrets. `off`, deterministic `fast`, and
/// resident `local_ai` are stable product values; optional cloud providers keep
/// their provider-qualified value so history never confuses them with local AI.
pub(crate) fn resolve_history_cleanup_mode(
    settings: &AppSettings,
    post_process: bool,
    deterministic_cleanup_applied: bool,
) -> String {
    if !post_process {
        return if deterministic_cleanup_applied {
            "fast".to_string()
        } else {
            "off".to_string()
        };
    }

    if settings.post_process_provider_id == LOCAL_CLEANUP_PROVIDER_ID {
        "local_ai".to_string()
    } else {
        format!("provider:{}", settings.post_process_provider_id)
    }
}

/// Resolve the model used by a history row to a stable engine-family identifier.
/// Unknown or already-removed models leave the field empty rather than guessing.
pub(crate) fn resolve_history_engine_type(
    app: &AppHandle,
    model_id: Option<&str>,
) -> Option<String> {
    let model_id = model_id?;
    let model = app.state::<Arc<ModelManager>>().get_model_info(model_id)?;
    let engine = match model.engine_type {
        EngineType::TranscribeCpp => "transcribe_cpp",
        EngineType::Parakeet => "parakeet",
        EngineType::Moonshine => "moonshine",
        EngineType::MoonshineStreaming => "moonshine_streaming",
        EngineType::SenseVoice => "sense_voice",
        EngineType::GigaAM => "gigaam",
        EngineType::Canary => "canary",
        EngineType::Cohere => "cohere",
    };
    Some(engine.to_string())
}

/// Map the exact transcribe.cpp load-time selection plan into history fields.
/// Non-transcribe.cpp engines leave these fields empty rather than attaching
/// unrelated Vulkan preferences to an ONNX history row.
pub(crate) fn resolve_history_compute_plan(
    engine_type: Option<&str>,
    selection_plan: Option<&crate::managers::transcription::TranscribeSelectionPlanMetadata>,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    if engine_type != Some("transcribe_cpp") {
        return (None, None, None, None);
    }

    let Some(selection_plan) = selection_plan else {
        return (None, None, None, None);
    };

    (
        selection_plan.saved_accelerator.clone(),
        selection_plan.saved_gpu_device.clone(),
        Some(selection_plan.recommended_backend.clone()),
        selection_plan.recommended_device.clone(),
    )
}

/// Resolve the persisted language *intent* into the language the currently-loaded
/// model will actually use — the same capability-aware coercion the transcription
/// paths apply (see [`crate::managers::model::effective_language`]). Post-processing
/// resolves it independently so it agrees with the language the transcription ran
/// in, without threading a value through the pipeline.
pub(crate) fn resolve_effective_language(app: &AppHandle, settings: &AppSettings) -> String {
    let tm = app.state::<Arc<TranscriptionManager>>();
    let model_manager = app.state::<Arc<ModelManager>>();
    let active_model = tm
        .get_current_model()
        .unwrap_or_else(|| settings.selected_model.clone());
    match model_manager.get_model_info(&active_model) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let cleanup_prompt_id = post_process
        .then(|| settings.post_process_selected_prompt_id.clone())
        .flatten();
    let cleanup_model_id = post_process
        .then(|| {
            settings
                .post_process_models
                .get(&settings.post_process_provider_id)
                .cloned()
        })
        .flatten()
        .filter(|model| !model.trim().is_empty());
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;
    let mut deterministic_cleanup_applied = false;

    // Resolve the language the transcription actually ran in (the persisted
    // intent coerced against the loaded model's capabilities) so OpenCC keys off
    // the effective language rather than a possibly-stale intent.
    let effective_language = resolve_effective_language(app, &settings);
    if let Some(converted_text) =
        maybe_convert_chinese_variant(&effective_language, transcription).await
    {
        deterministic_cleanup_applied = converted_text != transcription;
        final_text = converted_text;
    }

    if post_process {
        let cleanup_input = final_text.clone();
        let cleanup_result = post_process_transcription(&settings, &cleanup_input).await;
        (final_text, post_processed_text) = apply_cleanup_fail_open(&cleanup_input, cleanup_result);

        if post_processed_text.is_some() {
            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    let cleanup_mode =
        resolve_history_cleanup_mode(&settings, post_process, deterministic_cleanup_applied);

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
        cleanup_prompt_id,
        cleanup_model_id,
        cleanup_mode,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();
        let performance_manager = app.state::<Arc<PerformanceManager>>();
        let cold_start = !tm.is_model_loaded();

        // Load ASR model and VAD model in parallel
        let kickoff_started = Instant::now();
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });
        let kickoff_elapsed = kickoff_started.elapsed();

        let binding_id = binding_id.to_string();
        let tray_started = Instant::now();
        set_tray_state(app, TrayIconState::Recording);
        let tray_elapsed = tray_started.elapsed();

        // Get the microphone mode to determine audio feedback timing
        let plan_started = Instant::now();
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;

        let selected_model_info = app
            .state::<Arc<ModelManager>>()
            .get_model_info(&settings.selected_model);

        // Use the app-facing model capability as the single pre-recording source
        // for live streaming decisions. Unknown support is represented as false
        // until the model registry is updated by discovery or runtime load.
        let model_supports_streaming = selected_model_info
            .as_ref()
            .map(|m| m.supports_streaming)
            .unwrap_or(false);
        let vad_policy = if !settings.vad_enabled {
            VadPolicy::Disabled
        } else if model_supports_streaming {
            VadPolicy::Streaming
        } else {
            VadPolicy::Offline
        };
        let insertion_mode = resolve_session_insertion_mode(
            settings.insertion_mode,
            settings.experimental_enabled,
            self.post_process,
            model_supports_streaming,
            settings.vad_enabled,
        );
        if settings.insertion_mode == InsertionMode::LiveCommittedExperimental && self.post_process
        {
            warn!(
                "Live committed insertion is incompatible with whole-transcript AI cleanup; using preview-only insertion for this session"
            );
        } else if settings.insertion_mode == InsertionMode::LiveCommittedExperimental
            && !settings.vad_enabled
        {
            warn!(
                "Live committed insertion requires positive VAD speech evidence; using preview-only insertion while VAD is disabled"
            );
        }
        let performance_session_id = performance_manager.begin_session(
            start_time,
            PerformanceSessionMetadata {
                cold_start,
                model_id: (!settings.selected_model.is_empty())
                    .then_some(settings.selected_model.clone()),
                engine_type: resolve_history_engine_type(app, Some(&settings.selected_model)),
                language: Some(resolve_effective_language(app, &settings)),
                cleanup_mode: resolve_history_cleanup_mode(&settings, self.post_process),
                insertion_mode: insertion_mode_history_value(insertion_mode).to_string(),
            },
        );
        tm.begin_insertion_session(insertion_mode);
        if model_supports_streaming {
            tm.start_stream();
        }
        let plan_elapsed = plan_started.elapsed();

        // Sizing the overlay follows the same advertised capability. A model that
        // doesn't stream (or whose capability is not known yet) gets the compact
        // pill instead of an oversized transparent live window.
        let overlay_started = Instant::now();
        match settings.overlay_style {
            OverlayStyle::Live if model_supports_streaming => utils::show_streaming_overlay(app),
            OverlayStyle::Live | OverlayStyle::Minimal => show_recording_overlay(app),
            OverlayStyle::None => {} // show_overlay_state no-ops on None anyway
        }
        // Everything above runs before capture can begin, so each span here is
        // added keypress->capture latency.
        debug!(
            "start-path pre-recording steps: model_kickoff={:?} tray={:?} settings+stream_plan={:?} overlay={:?}",
            kickoff_elapsed,
            tray_elapsed,
            plan_elapsed,
            overlay_started.elapsed()
        );
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        let recording_start_time = Instant::now();
        match rm.try_start_recording(&binding_id, vad_policy) {
            Ok(readiness) => {
                // Queue media control only after microphone capture has been accepted.
                // The controller performs every platform call on its own worker, so
                // this cannot add media-service latency to the hotkey/capture path.
                app.state::<RecordingMediaController>()
                    .begin_recording(settings.pause_media_while_recording);
                debug!(
                    "Recording request accepted in {:?}; waiting for first microphone samples",
                    recording_start_time.elapsed()
                );
                let generation = readiness.generation();
                let app_clone = app.clone();
                let rm_clone = Arc::clone(&rm);
                let performance_manager = Arc::clone(&performance_manager);
                std::thread::spawn(move || {
                    if !readiness.wait() {
                        debug!("Microphone readiness wait ended without receiving samples");
                        return;
                    }

                    // Development-only preview hook for evaluating the brief
                    // arming animation on hardware that normally starts too fast
                    // to make it visible.
                    #[cfg(debug_assertions)]
                    if let Ok(delay_ms) = std::env::var("HANDY_DEBUG_MIC_READY_DELAY_MS")
                        .unwrap_or_default()
                        .parse::<u64>()
                    {
                        let delay_ms = delay_ms.min(10_000);
                        if delay_ms > 0 {
                            debug!("Delaying microphone-ready cue by {delay_ms}ms for UI preview");
                            std::thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }

                    if !rm_clone.is_recording_readiness_current(generation) {
                        debug!("Microphone became ready for an inactive recording");
                        return;
                    }

                    debug!("Microphone is receiving samples; recording is ready");
                    performance_manager.mark_capture_ready(performance_session_id);
                    utils::emit_recording_ready(&app_clone);

                    // The start chime is a readiness cue, so it must follow the
                    // first real input callback rather than Stream::play() or a
                    // fixed delay. The helper returns immediately when feedback
                    // is disabled; mute still follows the same readiness point.
                    if rm_clone.is_recording_readiness_current(generation) {
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                    }
                    if rm_clone.is_recording_readiness_current(generation) {
                        rm_clone.apply_mute();
                    }
                });
            }
            Err(e) => {
                debug!("Failed to start recording: {}", e);
                recording_error = Some(e);
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            tm.cancel_stream();
            performance_manager.finish_session(performance_session_id, "failure");
            utils::hide_recording_overlay(app);
            set_tray_state(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Prevent a slow microphone from emitting a ready event or start chime
        // after the user has already requested stop.
        app.state::<Arc<AudioRecordingManager>>()
            .invalidate_recording_readiness();
        // Resume controller-owned playback as soon as recording ends; do not wait
        // for transcription, cleanup, or paste to complete.
        app.state::<RecordingMediaController>().finish_recording();

        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());
        let performance_manager = Arc::clone(&app.state::<Arc<PerformanceManager>>());
        let performance_session_id = performance_manager.active_session_id();
        if let Some(session_id) = performance_session_id {
            performance_manager.mark_stop_requested(session_id);
        }

        set_tray_state(app, TrayIconState::Transcribing);
        // Stop should give immediate visual feedback. Live streaming can keep
        // the larger panel, but it still switches from listening to a working
        // spinner while the stream finalizes. Non-streaming paths use the
        // compact transcribing pill (None no-ops in show_*).
        let stop_settings = get_settings(app);
        let style = stop_settings.overlay_style;
        let session_insertion_mode = tm.current_insertion_mode();
        // Capture this before finalizing the stream so every later working state
        // targets the same overlay that was shown for this transcription.
        let use_streaming_overlay = should_use_streaming_overlay(style, tm.is_streaming());
        if use_streaming_overlay {
            tm.emit_stream_working(StreamWorkKind::Transcribing);
        } else {
            show_transcribing_overlay(app);
        }

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;
        let cancel_generation = rm.cancel_generation();

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if rm.was_cancelled_since(cancel_generation) {
                    debug!("Transcription operation cancelled after recording stop");
                    tm.cancel_stream();
                    if let Some(session_id) = performance_session_id {
                        performance_manager.finish_session(session_id, "cancelled");
                    }
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                    return;
                }

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    // Tear down any streaming worker so its channel doesn't leak
                    // and block the next start_stream.
                    tm.cancel_stream();
                    if let Some(session_id) = performance_session_id {
                        performance_manager.finish_session(session_id, "failure");
                    }
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save. If a live stream was
                    // running, finalize it and use its text (all audio was already
                    // fed to the stream); otherwise batch-transcribe the samples.
                    let transcription_time = Instant::now();
                    // Preserve the stream timing-only sample for diagnostics before
                    // reducing the finalized stream to the text-or-batch contract.
                    let stream_result = tm.finalize_stream_with_benchmark_timing();
                    let stream_timing = stream_result
                        .as_ref()
                        .ok()
                        .and_then(|result| result.as_ref().map(|(_, timing)| timing.clone()));
                    let used_stream_result = stream_result
                        .as_ref()
                        .ok()
                        .and_then(|result| result.as_ref())
                        .is_some_and(|(text, _)| !text.trim().is_empty());
                    let transcription_result = resolve_stream_or_batch(
                        stream_result.map(|result| result.map(|(text, _)| text)),
                        || tm.transcribe(samples),
                    );
                    let transcription_elapsed_ms = transcription_time.elapsed().as_millis() as u64;
                    let history_transcription_total_ms =
                        i64::try_from(transcription_elapsed_ms).unwrap_or(i64::MAX);
                    if let Some(session_id) = performance_session_id {
                        performance_manager.record_stage(
                            session_id,
                            "transcription_total",
                            transcription_elapsed_ms,
                        );
                        performance_manager.set_first_partial_ms(
                            session_id,
                            stream_timing
                                .as_ref()
                                .and_then(|timing| timing.first_partial_ms),
                        );
                        performance_manager.record_stage_since_stop(
                            session_id,
                            if used_stream_result {
                                "capture_stop_to_stream_finalize"
                            } else {
                                "capture_stop_to_batch_transcription"
                            },
                        );
                    }

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    let history_settings = get_settings(&ah);
                    let selected_history_model_id = (!history_settings.selected_model.is_empty())
                        .then_some(history_settings.selected_model.clone());
                    let history_language = resolve_effective_language(&ah, &history_settings);
                    let history_language =
                        (!history_language.is_empty()).then_some(history_language);
                    let history_duration_ms =
                        i64::try_from(sample_count.saturating_mul(1000) / 16_000)
                            .unwrap_or(i64::MAX);
                    let (history_backend, history_device, history_recovery_reason) =
                        tm.runtime_metadata();
                    if let Some(session_id) = performance_session_id {
                        performance_manager
                            .set_recording_ms(session_id, history_duration_ms.max(0) as u64);
                        performance_manager.update_runtime_metadata(
                            session_id,
                            history_backend.clone(),
                            history_device.clone(),
                        );
                    }
                    let history_selection_plan = tm.selection_plan_metadata();

                    if rm.was_cancelled_since(cancel_generation) {
                        debug!("Transcription operation cancelled before output handling");
                        if let Some(session_id) = performance_session_id {
                            performance_manager.finish_session(session_id, "cancelled");
                        }
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    }

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                utils::redact_text(&transcription)
                            );

                            if post_process {
                                if use_streaming_overlay {
                                    tm.emit_stream_working(StreamWorkKind::Polishing);
                                } else {
                                    show_processing_overlay(&ah);
                                }
                            }
                            let cleanup_time = Instant::now();
                            let Some(processed) = complete_unless_cancelled(
                                process_transcription_output(&ah, &transcription, post_process),
                                || rm.was_cancelled_since(cancel_generation),
                            )
                            .await
                            else {
                                debug!("Transcription operation cancelled during output handling");
                                if let Some(session_id) = performance_session_id {
                                    performance_manager.finish_session(session_id, "cancelled");
                                }
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            };
                            let history_cleanup_total_ms = post_process.then(|| {
                                i64::try_from(cleanup_time.elapsed().as_millis())
                                    .unwrap_or(i64::MAX)
                            });
                            if let (Some(session_id), Some(cleanup_ms)) =
                                (performance_session_id, history_cleanup_total_ms)
                            {
                                performance_manager.record_stage(
                                    session_id,
                                    "cleanup_total",
                                    cleanup_ms.max(0) as u64,
                                );
                            }

                            if rm.was_cancelled_since(cancel_generation) {
                                debug!("Transcription operation cancelled before paste");
                                if let Some(session_id) = performance_session_id {
                                    performance_manager.finish_session(session_id, "cancelled");
                                }
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            // Save to history if WAV was saved
                            if wav_saved {
                                let history_model_id = tm
                                    .get_current_model()
                                    .or_else(|| selected_history_model_id.clone());
                                let history_engine_type =
                                    resolve_history_engine_type(&ah, history_model_id.as_deref());
                                let (
                                    history_saved_accelerator,
                                    history_saved_gpu_device,
                                    history_recommended_backend,
                                    history_recommended_device,
                                ) = resolve_history_compute_plan(
                                    history_engine_type.as_deref(),
                                    history_selection_plan.as_ref(),
                                );
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                    processed.cleanup_prompt_id.clone(),
                                    processed.cleanup_model_id.clone(),
                                    history_model_id,
                                    history_engine_type,
                                    history_language.clone(),
                                    Some(
                                        insertion_mode_history_value(session_insertion_mode)
                                            .to_string(),
                                    ),
                                    history_backend.clone(),
                                    history_device.clone(),
                                    history_recovery_reason.clone(),
                                    history_saved_accelerator,
                                    history_saved_gpu_device,
                                    history_recommended_backend,
                                    history_recommended_device,
                                    Some(processed.cleanup_mode.clone()),
                                    Some("success".to_string()),
                                    Some(history_transcription_total_ms),
                                    history_cleanup_total_ms,
                                    history_duration_ms,
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            let block_whole_transcript_paste =
                                tm.live_insertion_blocks_final_paste();
                            if !should_paste_final_output(
                                processed.final_text.is_empty(),
                                session_insertion_mode,
                                block_whole_transcript_paste,
                            ) {
                                if block_whole_transcript_paste {
                                    debug!(
                                        "Skipping whole-transcript paste after live insertion/safety stop"
                                    );
                                }
                                if let Some(session_id) = performance_session_id {
                                    if !processed.final_text.is_empty() {
                                        performance_manager.mark_visible_text(session_id);
                                    }
                                    performance_manager.finish_session(session_id, "success");
                                }
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                let rm_for_paste = Arc::clone(&rm);
                                let performance_for_paste = Arc::clone(&performance_manager);
                                ah.run_on_main_thread(move || {
                                    if rm_for_paste.was_cancelled_since(cancel_generation) {
                                        debug!("Transcription operation cancelled before paste");
                                        if let Some(session_id) = performance_session_id {
                                            performance_for_paste
                                                .finish_session(session_id, "cancelled");
                                        }
                                        utils::hide_recording_overlay(&ah_clone);
                                        set_tray_state(&ah_clone, TrayIconState::Idle);
                                        return;
                                    }

                                    let paste_outcome =
                                        match utils::paste(final_text, ah_clone.clone()) {
                                            Ok(()) => {
                                                debug!(
                                                    "Text pasted successfully in {:?}",
                                                    paste_time.elapsed()
                                                );
                                                "success"
                                            }
                                            Err(e) => {
                                                error!("Failed to paste transcription: {}", e);
                                                let _ = ah_clone.emit("paste-error", ());
                                                "failure"
                                            }
                                        };
                                    if let Some(session_id) = performance_session_id {
                                        performance_for_paste.record_stage(
                                            session_id,
                                            "paste_total",
                                            paste_time.elapsed().as_millis() as u64,
                                        );
                                        performance_for_paste.mark_visible_text(session_id);
                                        performance_for_paste
                                            .finish_session(session_id, paste_outcome);
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    set_tray_state(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    if let Some(session_id) = performance_session_id {
                                        performance_manager.finish_session(session_id, "failure");
                                    }
                                    utils::hide_recording_overlay(&ah);
                                    set_tray_state(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            if rm.was_cancelled_since(cancel_generation) {
                                debug!(
                                    "Transcription operation cancelled after transcription error"
                                );
                                if let Some(session_id) = performance_session_id {
                                    performance_manager.finish_session(session_id, "cancelled");
                                }
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            error!("Transcription failed: {}", err);
                            // Surface the failure to the UI (toast). The full
                            // message is also in handy.log via the line above.
                            let _ = ah.emit("transcription-error", err.to_string());
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                let history_engine_type = resolve_history_engine_type(
                                    &ah,
                                    selected_history_model_id.as_deref(),
                                );
                                let (
                                    history_saved_accelerator,
                                    history_saved_gpu_device,
                                    history_recommended_backend,
                                    history_recommended_device,
                                ) = resolve_history_compute_plan(
                                    history_engine_type.as_deref(),
                                    history_selection_plan.as_ref(),
                                );
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                    None,
                                    None,
                                    selected_history_model_id,
                                    history_engine_type,
                                    history_language,
                                    Some(
                                        insertion_mode_history_value(session_insertion_mode)
                                            .to_string(),
                                    ),
                                    history_backend,
                                    history_device,
                                    history_recovery_reason,
                                    history_saved_accelerator,
                                    history_saved_gpu_device,
                                    history_recommended_backend,
                                    history_recommended_device,
                                    Some(resolve_history_cleanup_mode(
                                        &history_settings,
                                        post_process,
                                        false,
                                    )),
                                    Some("failure".to_string()),
                                    Some(history_transcription_total_ms),
                                    None,
                                    history_duration_ms,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            if let Some(session_id) = performance_session_id {
                                performance_manager.finish_session(session_id, "failure");
                            }
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                // Tear down any streaming worker so its channel doesn't leak.
                tm.cancel_stream();
                if let Some(session_id) = performance_session_id {
                    let outcome = if rm.was_cancelled_since(cancel_generation) {
                        "cancelled"
                    } else {
                        "failure"
                    };
                    performance_manager.finish_session(session_id, outcome);
                }
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::{
        apply_cleanup_fail_open, build_legacy_prompt, build_system_prompt,
        complete_unless_cancelled, insertion_mode_history_value, is_blank_transcription,
        normalize_cleanup_output, parse_structured_cleanup_output, resolve_history_cleanup_mode,
        resolve_history_compute_plan, resolve_session_insertion_mode, resolve_stream_or_batch,
        should_paste_final_output, should_use_streaming_overlay, strip_think_block,
        CLEANUP_MAX_OUTPUT_CHARS, CLEANUP_OUTPUT_CONTRACT,
    };
    use crate::settings::{AppSettings, InsertionMode, OverlayStyle};
    use std::cell::Cell;
    use std::future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn blank_transcription_is_detected() {
        assert!(is_blank_transcription(""));
        assert!(is_blank_transcription("   "));
        assert!(is_blank_transcription("\t\n  \r\n"));
    }

    #[test]
    fn non_blank_transcription_is_kept() {
        assert!(!is_blank_transcription("hello"));
        assert!(!is_blank_transcription("  hello  "));
    }

    #[test]
    fn history_cleanup_mode_preserves_off_fast_local_ai_and_cloud_semantics() {
        let cloud = AppSettings {
            post_process_provider_id: "openrouter".to_string(),
            ..Default::default()
        };
        let local = AppSettings {
            post_process_provider_id: crate::settings::LOCAL_CLEANUP_PROVIDER_ID.to_string(),
            ..Default::default()
        };

        assert_eq!(resolve_history_cleanup_mode(&cloud, false, false), "off");
        assert_eq!(resolve_history_cleanup_mode(&cloud, false, true), "fast");
        assert_eq!(
            resolve_history_cleanup_mode(&local, true, false),
            "local_ai"
        );
        assert_eq!(
            resolve_history_cleanup_mode(&cloud, true, false),
            "provider:openrouter"
        );
    }

    #[test]
    fn history_compute_plan_skips_unrelated_engine_families() {
        let plan = crate::managers::transcription::TranscribeSelectionPlanMetadata {
            saved_accelerator: Some("gpu".to_string()),
            saved_gpu_device: Some("stable-device-id".to_string()),
            recommended_backend: "auto".to_string(),
            recommended_device: Some("Discrete GPU".to_string()),
        };
        assert_eq!(
            resolve_history_compute_plan(Some("parakeet"), Some(&plan)),
            (None, None, None, None)
        );
    }

    #[test]
    fn history_compute_plan_preserves_exact_load_time_plan() {
        let plan = crate::managers::transcription::TranscribeSelectionPlanMetadata {
            saved_accelerator: Some("gpu".to_string()),
            saved_gpu_device: Some("stable-device-id".to_string()),
            recommended_backend: "auto".to_string(),
            recommended_device: Some("Discrete GPU".to_string()),
        };

        assert_eq!(
            resolve_history_compute_plan(Some("transcribe_cpp"), Some(&plan)),
            (
                Some("gpu".to_string()),
                Some("stable-device-id".to_string()),
                Some("auto".to_string()),
                Some("Discrete GPU".to_string()),
            )
        );
        assert_eq!(
            resolve_history_compute_plan(Some("transcribe_cpp"), None),
            (None, None, None, None)
        );
    }

    #[test]
    fn completed_operation_returns_its_output() {
        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::ready("done"),
            || false,
        ));

        assert_eq!(result, Some("done"));
    }

    #[test]
    fn pending_operation_stops_after_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_for_thread = Arc::clone(&cancelled);
        let cancel_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            cancelled_for_thread.store(true, Ordering::Release);
        });

        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::pending::<()>(),
            || cancelled.load(Ordering::Acquire),
        ));

        cancel_thread.join().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn leading_think_block_is_stripped() {
        assert_eq!(
            strip_think_block("<think>pondering...</think>Cleaned text."),
            "Cleaned text."
        );
        assert_eq!(
            strip_think_block("  \n<think>multi\nline</think>\n  Cleaned text."),
            "Cleaned text."
        );
    }

    #[test]
    fn content_without_think_block_is_unchanged() {
        assert_eq!(strip_think_block("Cleaned text."), "Cleaned text.");
        assert_eq!(
            strip_think_block("Mentions <think> mid-sentence."),
            "Mentions <think> mid-sentence."
        );
        // Unclosed block: leave untouched rather than guess
        assert_eq!(
            strip_think_block("<think>never closed"),
            "<think>never closed"
        );
    }

    #[test]
    fn cleanup_prompts_always_include_strict_output_contract() {
        let system = build_system_prompt("Fix punctuation for ${output}");
        assert!(!system.contains("${output}"));
        assert!(system.ends_with(CLEANUP_OUTPUT_CONTRACT));

        let legacy = build_legacy_prompt("Clean this: ${output}", "hello world");
        assert!(legacy.contains("Clean this: hello world"));
        assert!(legacy.ends_with(CLEANUP_OUTPUT_CONTRACT));
    }

    #[test]
    fn cleanup_output_accepts_plain_text_and_strips_reasoning_noise() {
        assert_eq!(
            normalize_cleanup_output("<think>ignore me</think>  Cleaned\u{200B} text.  "),
            Some("Cleaned text.".to_string())
        );
    }

    #[test]
    fn cleanup_output_rejects_empty_or_wrapped_responses() {
        assert_eq!(normalize_cleanup_output("   "), None);
        assert_eq!(
            normalize_cleanup_output("```text\nCleaned text.\n```"),
            None
        );
        assert_eq!(normalize_cleanup_output("\"Cleaned text.\""), None);
        assert_eq!(normalize_cleanup_output("'Cleaned text.'"), None);
        assert_eq!(normalize_cleanup_output("`Cleaned text.`"), None);
        assert_eq!(
            normalize_cleanup_output(r#"{"transcription":"Cleaned text."}"#),
            None
        );
        assert_eq!(
            normalize_cleanup_output("<transcription>Cleaned text.</transcription>"),
            None
        );
        assert_eq!(
            normalize_cleanup_output("Here is the cleaned transcription: Cleaned text."),
            None
        );
    }

    #[test]
    fn cleanup_output_rejects_text_beyond_the_hard_bound() {
        let oversized = "x".repeat(CLEANUP_MAX_OUTPUT_CHARS + 1);
        assert_eq!(normalize_cleanup_output(&oversized), None);
    }

    #[test]
    fn cleanup_failures_and_malformed_output_preserve_raw_text() {
        let raw = "raw transcript stays available";
        assert_eq!(apply_cleanup_fail_open(raw, None), (raw.to_string(), None));
        assert_eq!(
            apply_cleanup_fail_open(
                raw,
                normalize_cleanup_output(r#"{"transcription":"wrapped"}"#)
            ),
            (raw.to_string(), None)
        );
    }

    #[test]
    fn valid_cleanup_replaces_raw_text_without_losing_history_value() {
        let cleaned = "Cleaned transcript.".to_string();
        assert_eq!(
            apply_cleanup_fail_open("raw transcript", Some(cleaned.clone())),
            (cleaned.clone(), Some(cleaned))
        );
    }

    #[test]
    fn structured_cleanup_requires_exact_transcription_field() {
        assert_eq!(
            parse_structured_cleanup_output(r#"{"transcription":"Cleaned text."}"#),
            Some("Cleaned text.".to_string())
        );
        assert_eq!(parse_structured_cleanup_output("not json"), None);
        assert_eq!(
            parse_structured_cleanup_output(r#"{"message":"Cleaned text."}"#),
            None
        );
        assert_eq!(
            parse_structured_cleanup_output(
                r#"{"transcription":"Cleaned text.","explanation":"extra"}"#
            ),
            None
        );
        assert_eq!(
            parse_structured_cleanup_output(r#"{"transcription":"```text\\nwrapped\\n```"}"#),
            None
        );
    }

    #[test]
    fn live_overlay_uses_streaming_states_only_for_streaming_models() {
        assert!(should_use_streaming_overlay(OverlayStyle::Live, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::Live, false));
        assert!(!should_use_streaming_overlay(OverlayStyle::Minimal, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::None, true));
    }

    #[test]
    fn insertion_mode_defaults_and_history_values_are_stable() {
        assert_eq!(InsertionMode::default(), InsertionMode::AtStop);
        assert_eq!(
            insertion_mode_history_value(InsertionMode::AtStop),
            "at_stop"
        );
        assert_eq!(
            insertion_mode_history_value(InsertionMode::PreviewOnly),
            "preview_only"
        );
        assert_eq!(
            insertion_mode_history_value(InsertionMode::LiveCommittedExperimental),
            "live_committed_experimental"
        );
    }

    #[test]
    fn live_committed_mode_downgrades_to_preview_for_whole_transcript_cleanup() {
        assert_eq!(
            resolve_session_insertion_mode(
                InsertionMode::LiveCommittedExperimental,
                true,
                true,
                true,
                true
            ),
            InsertionMode::PreviewOnly
        );
        assert_eq!(
            resolve_session_insertion_mode(
                InsertionMode::LiveCommittedExperimental,
                true,
                false,
                true,
                true
            ),
            InsertionMode::LiveCommittedExperimental
        );
    }

    #[test]
    fn insertion_modes_require_the_experimental_master_switch() {
        assert_eq!(
            resolve_session_insertion_mode(
                InsertionMode::LiveCommittedExperimental,
                false,
                false,
                true,
                true
            ),
            InsertionMode::AtStop
        );
        assert_eq!(
            resolve_session_insertion_mode(InsertionMode::PreviewOnly, false, false, true, true),
            InsertionMode::AtStop
        );
    }

    #[test]
    fn live_committed_mode_downgrades_to_preview_without_positive_vad_evidence() {
        assert_eq!(
            resolve_session_insertion_mode(
                InsertionMode::LiveCommittedExperimental,
                true,
                false,
                true,
                false,
            ),
            InsertionMode::PreviewOnly
        );
        assert_eq!(
            resolve_session_insertion_mode(InsertionMode::PreviewOnly, true, false, true, false),
            InsertionMode::PreviewOnly
        );
    }

    #[test]
    fn insertion_modes_fall_back_to_at_stop_without_streaming_support() {
        for mode in [
            InsertionMode::AtStop,
            InsertionMode::PreviewOnly,
            InsertionMode::LiveCommittedExperimental,
        ] {
            assert_eq!(
                resolve_session_insertion_mode(mode, true, false, false, true),
                InsertionMode::AtStop
            );
        }
    }

    #[test]
    fn live_session_never_whole_repastes_after_delivery_or_safety_stop() {
        assert!(!should_paste_final_output(
            false,
            InsertionMode::LiveCommittedExperimental,
            true,
        ));
        assert!(should_paste_final_output(
            false,
            InsertionMode::LiveCommittedExperimental,
            false,
        ));
        assert!(should_paste_final_output(
            false,
            InsertionMode::PreviewOnly,
            true,
        ));
        assert!(!should_paste_final_output(
            true,
            InsertionMode::AtStop,
            false,
        ));
    }

    #[test]
    fn non_streaming_session_falls_back_to_batch_transcription() {
        let batch_called = Cell::new(false);
        let result: Result<String, &'static str> = resolve_stream_or_batch(Ok(None), || {
            batch_called.set(true);
            Ok("batch transcript".to_string())
        });

        assert_eq!(result.unwrap(), "batch transcript");
        assert!(batch_called.get());
    }
}
