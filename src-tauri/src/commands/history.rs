use crate::actions::{process_transcription_output, ProcessedTranscription};
use crate::managers::{
    history::{HistoryManager, PaginatedHistory},
    transcription::TranscriptionManager,
};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, State};

fn ensure_retry_transcription_audio(samples: &[f32]) -> Result<(), String> {
    if samples.is_empty() {
        Err("Recording has no audio samples".to_string())
    } else {
        Ok(())
    }
}

fn ensure_retry_transcription_text(transcription: &str) -> Result<(), String> {
    if transcription.trim().is_empty() {
        Err("Recording contains no speech".to_string())
    } else {
        Ok(())
    }
}

fn retry_cleanup_input(raw_text: &str) -> Result<&str, String> {
    if raw_text.trim().is_empty() {
        Err("Cannot retry cleanup without a raw transcription".to_string())
    } else {
        Ok(raw_text)
    }
}

fn retry_cleanup_update(
    processed: ProcessedTranscription,
) -> Result<(String, Option<String>, String), String> {
    let cleaned_text = processed
        .post_processed_text
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Cleanup did not produce updated text".to_string())?;

    Ok((
        cleaned_text,
        processed.post_process_prompt,
        processed.cleanup_mode,
    ))
}

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
    let path = history_manager
        .get_audio_file_path(&file_name)
        .map_err(|error| error.to_string())?;
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

    let audio_path = history_manager
        .get_audio_file_path(&entry.file_name)
        .map_err(|e| format!("Failed to locate retained audio: {e}"))?;
    let samples = crate::audio_toolkit::read_wav_samples(&audio_path)
        .map_err(|e| format!("Failed to load audio: {}", e))?;

    ensure_retry_transcription_audio(&samples)?;

    transcription_manager.initiate_model_load();

    let tm = Arc::clone(&transcription_manager);
    let transcription_started = Instant::now();
    let transcription = tauri::async_runtime::spawn_blocking(move || tm.transcribe(samples))
        .await
        .map_err(|e| format!("Transcription task panicked: {}", e))?
        .map_err(|e| e.to_string())?;
    let transcription_total_ms =
        i64::try_from(transcription_started.elapsed().as_millis()).unwrap_or(i64::MAX);

    ensure_retry_transcription_text(&transcription)?;

    let history_settings = crate::settings::get_settings(&app);
    let model_id = transcription_manager.get_current_model().or_else(|| {
        (!history_settings.selected_model.is_empty())
            .then_some(history_settings.selected_model.clone())
    });
    let engine_type = crate::actions::resolve_history_engine_type(&app, model_id.as_deref());
    let selection_plan = transcription_manager.selection_plan_metadata();
    let (saved_accelerator, saved_gpu_device, recommended_backend, recommended_device) =
        crate::actions::resolve_history_compute_plan(
            engine_type.as_deref(),
            selection_plan.as_ref(),
        );
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
            saved_accelerator,
            saved_gpu_device,
            recommended_backend,
            recommended_device,
            Some(processed.cleanup_mode.clone()),
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

    let raw_text = retry_cleanup_input(&entry.transcription_text)?;

    let cleanup_started = Instant::now();
    let processed = process_transcription_output(&app, raw_text, true).await;
    let cleanup_total_ms = i64::try_from(cleanup_started.elapsed().as_millis()).unwrap_or(i64::MAX);
    let (cleaned_text, post_process_prompt, cleanup_mode) = retry_cleanup_update(processed)?;

    history_manager
        .update_cleanup(
            id,
            cleaned_text,
            post_process_prompt,
            Some(cleanup_mode),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn processed_cleanup(text: Option<&str>) -> ProcessedTranscription {
        ProcessedTranscription {
            final_text: text.unwrap_or("raw transcript").to_string(),
            post_processed_text: text.map(str::to_string),
            post_process_prompt: Some("cleanup prompt".to_string()),
            cleanup_mode: "provider:openai".to_string(),
        }
    }

    #[test]
    fn retry_transcription_requires_retained_audio() {
        assert_eq!(
            ensure_retry_transcription_audio(&[]).unwrap_err(),
            "Recording has no audio samples"
        );
        assert!(ensure_retry_transcription_audio(&[0.25]).is_ok());
    }

    #[test]
    fn retry_transcription_rejects_empty_or_whitespace_only_results() {
        for text in ["", "   ", "\n\t"] {
            assert_eq!(
                ensure_retry_transcription_text(text).unwrap_err(),
                "Recording contains no speech"
            );
        }
        assert!(ensure_retry_transcription_text("new raw transcript").is_ok());
    }

    #[test]
    fn retry_cleanup_uses_stored_raw_text_without_audio_input() {
        let raw = "stored raw transcript";
        assert_eq!(retry_cleanup_input(raw).unwrap(), raw);
        assert_eq!(
            retry_cleanup_input("  ").unwrap_err(),
            "Cannot retry cleanup without a raw transcription"
        );
    }

    #[test]
    fn retry_cleanup_requires_a_real_cleaned_result_before_persisting() {
        assert_eq!(
            retry_cleanup_update(processed_cleanup(None)).unwrap_err(),
            "Cleanup did not produce updated text"
        );

        let (text, prompt, mode) = retry_cleanup_update(processed_cleanup(Some("cleaned text")))
            .expect("valid cleanup should be persisted");
        assert_eq!(text, "cleaned text");
        assert_eq!(prompt.as_deref(), Some("cleanup prompt"));
        assert_eq!(mode, "provider:openai");
    }
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
