use serde::Serialize;
use specta::Type;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const PERFORMANCE_SAMPLE_LIMIT: usize = 200;

#[derive(Clone, Debug, Serialize, Type, PartialEq, Eq)]
pub struct PerformanceStage {
    pub name: String,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, Serialize, Type, PartialEq, Eq)]
pub struct PerformanceSample {
    pub session_id: u64,
    pub outcome: String,
    pub cold_start: bool,
    pub model_id: Option<String>,
    pub engine_type: Option<String>,
    pub language: Option<String>,
    pub backend: Option<String>,
    pub device: Option<String>,
    pub cleanup_mode: String,
    pub insertion_mode: String,
    pub recording_ms: Option<u64>,
    pub first_partial_ms: Option<u64>,
    pub stages: Vec<PerformanceStage>,
}

#[derive(Clone, Debug)]
pub struct PerformanceSessionMetadata {
    pub cold_start: bool,
    pub model_id: Option<String>,
    pub engine_type: Option<String>,
    pub language: Option<String>,
    pub cleanup_mode: String,
    pub insertion_mode: String,
}

#[derive(Clone, Debug, Serialize, Type, PartialEq, Eq)]
pub struct StagePercentiles {
    pub stage: String,
    pub sample_count: usize,
    pub p50_ms: u64,
    pub p95_ms: u64,
}

#[derive(Clone, Debug, Serialize, Type, PartialEq, Eq)]
pub struct PerformanceWindowSummary {
    pub window: usize,
    pub sample_count: usize,
    pub stages: Vec<StagePercentiles>,
}

#[derive(Clone, Debug, Serialize, Type, PartialEq, Eq)]
pub struct PerformanceSnapshot {
    pub sample_count: usize,
    pub latest: Option<PerformanceSample>,
    pub windows: Vec<PerformanceWindowSummary>,
}

struct ActivePerformanceSession {
    sample: PerformanceSample,
    started_at: Instant,
    capture_ready_at: Option<Instant>,
    stop_requested_at: Option<Instant>,
}

impl ActivePerformanceSession {
    fn new(session_id: u64, started_at: Instant, metadata: PerformanceSessionMetadata) -> Self {
        Self {
            sample: PerformanceSample {
                session_id,
                outcome: "running".to_string(),
                cold_start: metadata.cold_start,
                model_id: metadata.model_id,
                engine_type: metadata.engine_type,
                language: metadata.language,
                backend: None,
                device: None,
                cleanup_mode: metadata.cleanup_mode,
                insertion_mode: metadata.insertion_mode,
                recording_ms: None,
                first_partial_ms: None,
                stages: Vec::new(),
            },
            started_at,
            capture_ready_at: None,
            stop_requested_at: None,
        }
    }

    fn set_stage(&mut self, name: &str, duration_ms: u64) {
        if let Some(stage) = self
            .sample
            .stages
            .iter_mut()
            .find(|stage| stage.name == name)
        {
            stage.duration_ms = duration_ms;
        } else {
            self.sample.stages.push(PerformanceStage {
                name: name.to_string(),
                duration_ms,
            });
        }
    }

    fn finish(mut self, outcome: &str, finished_at: Instant) -> PerformanceSample {
        if let Some(stop_requested_at) = self.stop_requested_at {
            self.set_stage(
                "stop_to_idle",
                duration_ms_between(stop_requested_at, finished_at),
            );
        }
        self.set_stage(
            "total_hotkey_to_idle",
            duration_ms_between(self.started_at, finished_at),
        );
        self.sample.outcome = outcome.to_string();
        self.sample
    }
}

pub struct PerformanceManager {
    next_session_id: AtomicU64,
    active: Mutex<Option<ActivePerformanceSession>>,
    samples: Mutex<VecDeque<PerformanceSample>>,
}

impl Default for PerformanceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceManager {
    pub fn new() -> Self {
        Self {
            next_session_id: AtomicU64::new(1),
            active: Mutex::new(None),
            samples: Mutex::new(VecDeque::with_capacity(PERFORMANCE_SAMPLE_LIMIT)),
        }
    }

    pub fn begin_session(&self, started_at: Instant, metadata: PerformanceSessionMetadata) -> u64 {
        let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let previous = self
            .active
            .lock()
            .unwrap()
            .replace(ActivePerformanceSession::new(
                session_id, started_at, metadata,
            ));
        if let Some(previous) = previous {
            self.push_sample(previous.finish("cancelled", Instant::now()));
        }
        session_id
    }

    pub fn active_session_id(&self) -> Option<u64> {
        self.active
            .lock()
            .unwrap()
            .as_ref()
            .map(|active| active.sample.session_id)
    }

    pub fn mark_capture_ready(&self, session_id: u64) {
        self.with_active(session_id, |active| {
            let now = Instant::now();
            active.capture_ready_at = Some(now);
            active.set_stage(
                "hotkey_to_capture_ready",
                duration_ms_between(active.started_at, now),
            );
        });
    }

    pub fn mark_stop_requested(&self, session_id: u64) {
        self.with_active(session_id, |active| {
            active.stop_requested_at.get_or_insert_with(Instant::now);
        });
    }

    pub fn set_recording_ms(&self, session_id: u64, recording_ms: u64) {
        self.with_active(session_id, |active| {
            active.sample.recording_ms = Some(recording_ms);
            active.set_stage("capture_duration", recording_ms);
        });
    }

    pub fn record_stage(&self, session_id: u64, name: &str, duration_ms: u64) {
        self.with_active(session_id, |active| active.set_stage(name, duration_ms));
    }

    pub fn record_stage_since_stop(&self, session_id: u64, name: &str) {
        self.with_active(session_id, |active| {
            if let Some(stop_requested_at) = active.stop_requested_at {
                active.set_stage(name, duration_ms_between(stop_requested_at, Instant::now()));
            }
        });
    }

    pub fn set_first_partial_ms(&self, session_id: u64, first_partial_ms: Option<u64>) {
        let Some(first_partial_ms) = first_partial_ms else {
            return;
        };
        self.with_active(session_id, |active| {
            active.sample.first_partial_ms = Some(first_partial_ms);
            active.set_stage("first_partial", first_partial_ms);
        });
    }

    pub fn update_runtime_metadata(
        &self,
        session_id: u64,
        backend: Option<String>,
        device: Option<String>,
    ) {
        self.with_active(session_id, |active| {
            active.sample.backend = backend;
            active.sample.device = device;
        });
    }

    pub fn mark_visible_text(&self, session_id: u64) {
        self.with_active(session_id, |active| {
            if let Some(stop_requested_at) = active.stop_requested_at {
                active.set_stage(
                    "stop_to_visible_text",
                    duration_ms_between(stop_requested_at, Instant::now()),
                );
            }
        });
    }

    pub fn finish_session(&self, session_id: u64, outcome: &str) -> bool {
        let active = {
            let mut guard = self.active.lock().unwrap();
            if guard
                .as_ref()
                .is_none_or(|active| active.sample.session_id != session_id)
            {
                return false;
            }
            guard.take().unwrap()
        };
        self.push_sample(active.finish(outcome, Instant::now()));
        true
    }

    pub fn finish_active(&self, outcome: &str) -> bool {
        let active = self.active.lock().unwrap().take();
        let Some(active) = active else {
            return false;
        };
        self.push_sample(active.finish(outcome, Instant::now()));
        true
    }

    pub fn clear(&self) {
        self.samples.lock().unwrap().clear();
    }

    pub fn snapshot(&self) -> PerformanceSnapshot {
        let samples = self.samples.lock().unwrap();
        snapshot_from_samples(&samples)
    }

    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.snapshot())
    }

    fn with_active(&self, session_id: u64, update: impl FnOnce(&mut ActivePerformanceSession)) {
        let mut active = self.active.lock().unwrap();
        let Some(active) = active.as_mut() else {
            return;
        };
        if active.sample.session_id == session_id {
            update(active);
        }
    }

    fn push_sample(&self, sample: PerformanceSample) {
        let mut samples = self.samples.lock().unwrap();
        if samples.len() == PERFORMANCE_SAMPLE_LIMIT {
            samples.pop_front();
        }
        samples.push_back(sample);
    }
}

fn duration_ms_between(start: Instant, end: Instant) -> u64 {
    end.saturating_duration_since(start).as_millis() as u64
}

fn snapshot_from_samples(samples: &VecDeque<PerformanceSample>) -> PerformanceSnapshot {
    PerformanceSnapshot {
        sample_count: samples.len(),
        latest: samples.back().cloned(),
        windows: [10usize, 50, 200]
            .into_iter()
            .map(|window| summarize_window(samples, window))
            .collect(),
    }
}

fn summarize_window(
    samples: &VecDeque<PerformanceSample>,
    window: usize,
) -> PerformanceWindowSummary {
    let start = samples.len().saturating_sub(window);
    let selected = samples.iter().skip(start).collect::<Vec<_>>();
    let mut stage_values: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for sample in &selected {
        for stage in &sample.stages {
            stage_values
                .entry(stage.name.as_str())
                .or_default()
                .push(stage.duration_ms);
        }
    }

    let stages = stage_values
        .into_iter()
        .filter_map(|(stage, values)| {
            let p50_ms = percentile(&values, 50)?;
            let p95_ms = percentile(&values, 95)?;
            Some(StagePercentiles {
                stage: stage.to_string(),
                sample_count: values.len(),
                p50_ms,
                p95_ms,
            })
        })
        .collect();

    PerformanceWindowSummary {
        window,
        sample_count: selected.len(),
        stages,
    }
}

fn percentile(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (percentile.saturating_mul(sorted.len()) + 99) / 100;
    Some(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(cold_start: bool) -> PerformanceSessionMetadata {
        PerformanceSessionMetadata {
            cold_start,
            model_id: Some("model-safe-id".to_string()),
            engine_type: Some("transcribe_cpp".to_string()),
            language: Some("en".to_string()),
            cleanup_mode: "off".to_string(),
            insertion_mode: "at_stop".to_string(),
        }
    }

    fn finish_sample(manager: &PerformanceManager, stage_ms: u64, outcome: &str) {
        let id = manager.begin_session(Instant::now(), metadata(false));
        manager.record_stage(id, "transcription_total", stage_ms);
        assert!(manager.finish_session(id, outcome));
    }

    #[test]
    fn retention_is_bounded_to_last_two_hundred_sessions() {
        let manager = PerformanceManager::new();
        for index in 0..(PERFORMANCE_SAMPLE_LIMIT + 7) {
            finish_sample(&manager, index as u64, "success");
        }

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.sample_count, PERFORMANCE_SAMPLE_LIMIT);
        assert_eq!(snapshot.latest.unwrap().session_id, 207);
        let window = snapshot
            .windows
            .iter()
            .find(|window| window.window == 200)
            .unwrap();
        assert_eq!(window.sample_count, PERFORMANCE_SAMPLE_LIMIT);
        let transcription = window
            .stages
            .iter()
            .find(|stage| stage.stage == "transcription_total")
            .unwrap();
        assert_eq!(transcription.sample_count, PERFORMANCE_SAMPLE_LIMIT);
    }

    #[test]
    fn percentiles_use_the_requested_recent_window() {
        let manager = PerformanceManager::new();
        for value in 1..=20 {
            finish_sample(&manager, value * 10, "success");
        }

        let snapshot = manager.snapshot();
        let last_ten = snapshot
            .windows
            .iter()
            .find(|window| window.window == 10)
            .unwrap();
        let stage = last_ten
            .stages
            .iter()
            .find(|stage| stage.stage == "transcription_total")
            .unwrap();
        assert_eq!(stage.sample_count, 10);
        assert_eq!(stage.p50_ms, 150);
        assert_eq!(stage.p95_ms, 200);
    }

    #[test]
    fn failed_and_cancelled_sessions_are_retained_as_outcomes() {
        let manager = PerformanceManager::new();
        finish_sample(&manager, 12, "failure");
        finish_sample(&manager, 8, "cancelled");

        let samples = manager.samples.lock().unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].outcome, "failure");
        assert_eq!(samples[1].outcome, "cancelled");
    }

    #[test]
    fn export_schema_has_no_content_bearing_fields() {
        fn collect_keys(value: &serde_json::Value, keys: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(object) => {
                    for (key, value) in object {
                        keys.push(key.to_ascii_lowercase());
                        collect_keys(value, keys);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        collect_keys(value, keys);
                    }
                }
                _ => {}
            }
        }

        let manager = PerformanceManager::new();
        let id = manager.begin_session(Instant::now(), metadata(true));
        manager.set_recording_ms(id, 1234);
        manager.set_first_partial_ms(id, Some(80));
        manager.record_stage(id, "transcription_total", 410);
        manager.update_runtime_metadata(id, Some("vulkan".to_string()), Some("gpu-0".to_string()));
        assert!(manager.finish_session(id, "success"));

        let export = manager.export_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&export).unwrap();
        let mut keys = Vec::new();
        collect_keys(&value, &mut keys);
        for forbidden in [
            "transcript",
            "transcription_text",
            "audio",
            "audio_data",
            "audio_path",
            "clipboard",
            "window_title",
            "process_path",
        ] {
            assert!(
                !keys.iter().any(|key| key == forbidden),
                "export leaked forbidden field {forbidden}"
            );
        }
        assert!(keys.iter().any(|key| key == "first_partial_ms"));
        assert!(keys.iter().any(|key| key == "backend"));
        assert!(keys.iter().any(|key| key == "cold_start"));
        assert!(export.contains("transcription_total"));
    }

    #[test]
    fn clear_affects_only_the_performance_ring() {
        let manager = PerformanceManager::new();
        finish_sample(&manager, 50, "success");
        assert_eq!(manager.snapshot().sample_count, 1);
        manager.clear();
        assert_eq!(manager.snapshot().sample_count, 0);
    }
}
