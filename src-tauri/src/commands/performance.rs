use crate::managers::performance::{PerformanceManager, PerformanceSnapshot};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
#[specta::specta]
pub fn get_performance_diagnostics(
    manager: State<'_, Arc<PerformanceManager>>,
) -> PerformanceSnapshot {
    manager.snapshot()
}

#[tauri::command]
#[specta::specta]
pub fn export_performance_diagnostics(
    manager: State<'_, Arc<PerformanceManager>>,
) -> Result<String, String> {
    manager.export_json().map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn clear_performance_diagnostics(manager: State<'_, Arc<PerformanceManager>>) {
    manager.clear();
}
