use crate::actions::process_transcription_output;
use crate::managers::{
    history::{HistoryManager, PaginatedHistory},
    transcription::TranscriptionManager,
};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, State};

#[tauri::command]
#[specta::specta]
pub async fn get_history_entries(
    _app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    cursor: Option<i64>,
    limit: Option<usize>,
    search: Option<String>,
    start_timestamp: Option<i64>,
    end_timestamp_exclusive: Option<i64>,
    model_filter: Option<String>,
    outcome_filter: Option<String>,
    cleanup_filter: Option<String>,
) -> Result<PaginatedHistory, String> {
    history_manager
        .get_history_entries(
            cursor,
            limit,
            search.as_deref(),
            start_timestamp,
            end_timestamp_exclusive,
            model_filter.as_deref(),
            outcome_filter.as_deref(),
            cleanup_filter.as_deref(),
        )
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn toggle_history_entry_saved(
    _app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    id: i64,
) -> Result<(), String> {
    history_manager
        .toggle_saved_status(id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn get_audio_file_path(
    _app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    file_name: String,
) -> Result<String, String> {
    let path = history_manager.get_audio_file_path(&file_name);
    path.to_str()
        .ok_or_else(|| "Invalid file path".to_string())
        .map(|s| s.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_history_entry(
    _app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    id: i64,
) -> Result<(), String> {
    history_manager
        .delete_entry(id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn retry_history_entry_transcription(
    app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    id: i64,
) -> Result<(), String> {
    let entry = history_manager
        .get_entry_by_id(id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("History entry {} not found", id))?;

    let audio_path = history_manager.get_audio_file_path(&entry.file_name);
    let samples = crate::audio_toolkit::read_wav_samples(&audio_path)
        .map_err(|e| format!("Failed to load audio: {}", e))?;

    if samples.is_empty() {
        return Err("Recording has no audio samples".to_string());
    }

    transcription_manager.initiate_model_load();

    let tm = Arc::clone(&transcription_manager);
    let transcription_started = Instant::now();
    let transcription = tauri::async_runtime::spawn_blocking(move || tm.transcribe(samples))
        .await
        .map_err(|e| format!("Transcription task panicked: {}", e))?
        .map_err(|e| e.to_string())?;
    let transcription_total_ms =
        i64::try_from(transcription_started.elapsed().as_millis()).unwrap_or(i64::MAX);

    if transcription.is_empty() {
        return Err("Recording contains no speech".to_string());
    }

    let history_settings = crate::settings::get_settings(&app);
    let model_id = transcription_manager.get_current_model().or_else(|| {
        (!history_settings.selected_model.is_empty())
            .then_some(history_settings.selected_model.clone())
    });
    let engine_type = crate::actions::resolve_history_engine_type(&app, model_id.as_deref());
    let language = crate::actions::resolve_effective_language(&app, &history_settings);
    let language = (!language.is_empty()).then_some(language);
    let backend = transcription_manager.current_backend();
    let device = transcription_manager.current_device();
    let cleanup_started = Instant::now();
    let processed =
        process_transcription_output(&app, &transcription, entry.post_process_requested).await;
    let cleanup_total_ms = entry
        .post_process_requested
        .then(|| i64::try_from(cleanup_started.elapsed().as_millis()).unwrap_or(i64::MAX));
    history_manager
        .update_transcription(
            id,
            transcription,
            processed.post_processed_text,
            processed.post_process_prompt,
            model_id,
            engine_type,
            language,
            backend,
            device,
            Some(transcription_total_ms),
            cleanup_total_ms,
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn retry_history_entry_cleanup(
    app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    id: i64,
) -> Result<(), String> {
    let entry = history_manager
        .get_entry_by_id(id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("History entry {} not found", id))?;

    if entry.transcription_text.trim().is_empty() {
        return Err("Cannot retry cleanup without a raw transcription".to_string());
    }

    let cleanup_started = Instant::now();
    let processed = process_transcription_output(&app, &entry.transcription_text, true).await;
    let cleanup_total_ms = i64::try_from(cleanup_started.elapsed().as_millis()).unwrap_or(i64::MAX);
    let cleaned_text = processed
        .post_processed_text
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Cleanup did not produce updated text".to_string())?;

    history_manager
        .update_cleanup(
            id,
            cleaned_text,
            processed.post_process_prompt,
            Some(cleanup_total_ms),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn update_history_limit(
    app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    limit: usize,
) -> Result<(), String> {
    let mut settings = crate::settings::get_settings(&app);
    settings.history_limit = limit;
    crate::settings::write_settings(&app, settings);

    history_manager
        .cleanup_old_entries()
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn update_recording_retention_period(
    app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    period: String,
) -> Result<(), String> {
    use crate::settings::RecordingRetentionPeriod;

    let retention_period = match period.as_str() {
        "never" => RecordingRetentionPeriod::Never,
        "preserve_limit" => RecordingRetentionPeriod::PreserveLimit,
        "days3" => RecordingRetentionPeriod::Days3,
        "weeks2" => RecordingRetentionPeriod::Weeks2,
        "months3" => RecordingRetentionPeriod::Months3,
        _ => return Err(format!("Invalid retention period: {}", period)),
    };

    let mut settings = crate::settings::get_settings(&app);
    settings.recording_retention_period = retention_period;
    crate::settings::write_settings(&app, settings);

    history_manager
        .cleanup_old_entries()
        .map_err(|e| e.to_string())?;

    Ok(())
}
