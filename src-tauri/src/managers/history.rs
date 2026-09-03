use anyhow::{anyhow, Result};
use chrono::{DateTime, Local, Utc};
use log::{debug, error, info};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use tauri::AppHandle;
use tauri_specta::Event;

/// Database migrations for transcription history.
/// Each migration is applied in order. The library tracks which migrations
/// have been applied using SQLite's user_version pragma.
///
/// Note: For users upgrading from tauri-plugin-sql, migrate_from_tauri_plugin_sql()
/// converts the old _sqlx_migrations table tracking to the user_version pragma,
/// ensuring migrations don't re-run on existing databases.
static MIGRATIONS: &[M] = &[
    M::up(
        "CREATE TABLE IF NOT EXISTS transcription_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            saved BOOLEAN NOT NULL DEFAULT 0,
            title TEXT NOT NULL,
            transcription_text TEXT NOT NULL
        );",
    ),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_processed_text TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_prompt TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_requested BOOLEAN NOT NULL DEFAULT 0;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN model_id TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN duration_ms INTEGER;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN language TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN engine_type TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN insertion_mode TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN backend TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN device TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN outcome TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN transcription_total_ms INTEGER;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN cleanup_total_ms INTEGER;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN cleanup_mode TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN saved_accelerator TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN saved_gpu_device TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN recommended_backend TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN recommended_device TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN cleanup_prompt_id TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN cleanup_model_id TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN recovery_reason TEXT;"),
];

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct PaginatedHistory {
    pub entries: Vec<HistoryEntry>,
    pub has_more: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
#[serde(tag = "action")]
pub enum HistoryUpdatePayload {
    #[serde(rename = "added")]
    Added { entry: HistoryEntry },
    #[serde(rename = "updated")]
    Updated { entry: HistoryEntry },
    #[serde(rename = "deleted")]
    Deleted { id: i64 },
    #[serde(rename = "toggled")]
    Toggled { id: i64 },
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct HistoryEntry {
    pub id: i64,
    pub file_name: String,
    pub timestamp: i64,
    pub duration_ms: Option<i64>,
    pub saved: bool,
    pub title: String,
    pub transcription_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    /// Stable cleanup prompt identifier. Kept separate from the prompt body so
    /// history can identify the exact configured cleanup contract without
    /// duplicating transcript/application context.
    pub cleanup_prompt_id: Option<String>,
    /// Cleanup model identifier; `model_id` below remains the ASR model.
    pub cleanup_model_id: Option<String>,
    pub post_process_requested: bool,
    pub model_id: Option<String>,
    /// Stable engine family identifier for the model used by this run.
    pub engine_type: Option<String>,
    /// Effective language mode used for the transcription. `auto` means the
    /// model was allowed to detect the language rather than a forced code.
    pub language: Option<String>,
    /// Text insertion behavior used by the original recording session.
    pub insertion_mode: Option<String>,
    /// Actual runtime compute backend used by the transcription engine.
    pub backend: Option<String>,
    /// Actual runtime compute device used by the transcription engine.
    pub device: Option<String>,
    /// Stable reason code when acceleration was downgraded for this session/run.
    pub recovery_reason: Option<String>,
    /// Saved transcribe.cpp accelerator preference for this session.
    pub saved_accelerator: Option<String>,
    /// Stable saved GPU device identity when the preference selected one exactly.
    pub saved_gpu_device: Option<String>,
    /// Backend requested by Handy's load recommendation before runtime fallback.
    pub recommended_backend: Option<String>,
    /// Readable device selected by the recommendation, when it pinned one exactly.
    pub recommended_device: Option<String>,
    /// Cleanup behavior used for this row: `off` or the selected provider path.
    pub cleanup_mode: Option<String>,
    /// Persisted session result. Current recording sessions use `success` or `failure`.
    pub outcome: Option<String>,
    /// Safe transcription-stage timing summary for this history row.
    pub transcription_total_ms: Option<i64>,
    /// Safe cleanup-stage timing summary when cleanup was requested for this row.
    pub cleanup_total_ms: Option<i64>,
}

pub struct HistoryManager {
    app_handle: AppHandle,
    recordings_dir: PathBuf,
    db_path: PathBuf,
}

impl HistoryManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        // Create recordings directory in app data dir
        let app_data_dir = crate::portable::app_data_dir(app_handle)?;
        let recordings_dir = app_data_dir.join("recordings");
        let db_path = app_data_dir.join("history.db");

        // Ensure recordings directory exists
        if !recordings_dir.exists() {
            fs::create_dir_all(&recordings_dir)?;
            debug!("Created recordings directory: {:?}", recordings_dir);
        }

        let manager = Self {
            app_handle: app_handle.clone(),
            recordings_dir,
            db_path,
        };

        // Initialize database and run migrations synchronously.
        manager.init_database()?;

        // Retention is maintenance, not a startup prerequisite. Run it here so
        // crash leftovers are eventually reclaimed, but never make a cleanup
        // filesystem problem prevent Handy from opening its history database.
        if let Err(error) = manager.cleanup_old_entries() {
            error!("History audio cleanup at startup failed: {error}");
        }

        Ok(manager)
    }

    fn init_database(&self) -> Result<()> {
        info!("Initializing database at {:?}", self.db_path);

        let mut conn = Connection::open(&self.db_path)?;

        // Handle migration from tauri-plugin-sql to rusqlite_migration
        // tauri-plugin-sql used _sqlx_migrations table, rusqlite_migration uses user_version pragma
        self.migrate_from_tauri_plugin_sql(&conn)?;

        // Create migrations object and run to latest version
        let migrations = Migrations::new(MIGRATIONS.to_vec());

        // Validate migrations in debug builds
        #[cfg(debug_assertions)]
        migrations.validate().expect("Invalid migrations");

        // Get current version before migration
        let version_before: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
        debug!("Database version before migration: {}", version_before);

        // Apply any pending migrations
        migrations.to_latest(&mut conn)?;

        // Get version after migration
        let version_after: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version_after > version_before {
            info!(
                "Database migrated from version {} to {}",
                version_before, version_after
            );
        } else {
            debug!("Database already at latest version {}", version_after);
        }

        Ok(())
    }

    /// Migrate from tauri-plugin-sql's migration tracking to rusqlite_migration's.
    /// tauri-plugin-sql used a _sqlx_migrations table, while rusqlite_migration uses
    /// SQLite's user_version pragma. This function checks if the old system was in use
    /// and sets the user_version accordingly so migrations don't re-run.
    fn migrate_from_tauri_plugin_sql(&self, conn: &Connection) -> Result<()> {
        // Check if the old _sqlx_migrations table exists
        let has_sqlx_migrations: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='_sqlx_migrations'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !has_sqlx_migrations {
            return Ok(());
        }

        // Check current user_version
        let current_version: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version > 0 {
            // Already migrated to rusqlite_migration system
            return Ok(());
        }

        // Get the highest version from the old migrations table
        let old_version: i32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM _sqlx_migrations WHERE success = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if old_version > 0 {
            info!(
                "Migrating from tauri-plugin-sql (version {}) to rusqlite_migration",
                old_version
            );

            // Set user_version to match the old migration state
            conn.pragma_update(None, "user_version", old_version)?;

            // Optionally drop the old migrations table (keeping it doesn't hurt)
            // conn.execute("DROP TABLE IF EXISTS _sqlx_migrations", [])?;

            info!(
                "Migration tracking converted: user_version set to {}",
                old_version
            );
        }

        Ok(())
    }

    fn get_connection(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    fn map_history_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<HistoryEntry> {
        Ok(HistoryEntry {
            id: row.get("id")?,
            file_name: row.get("file_name")?,
            timestamp: row.get("timestamp")?,
            duration_ms: row.get("duration_ms")?,
            saved: row.get("saved")?,
            title: row.get("title")?,
            transcription_text: row.get("transcription_text")?,
            post_processed_text: row.get("post_processed_text")?,
            post_process_prompt: row.get("post_process_prompt")?,
            cleanup_prompt_id: row.get("cleanup_prompt_id")?,
            cleanup_model_id: row.get("cleanup_model_id")?,
            post_process_requested: row.get("post_process_requested")?,
            model_id: row.get("model_id")?,
            engine_type: row.get("engine_type")?,
            language: row.get("language")?,
            insertion_mode: row.get("insertion_mode")?,
            backend: row.get("backend")?,
            device: row.get("device")?,
            recovery_reason: row.get("recovery_reason")?,
            saved_accelerator: row.get("saved_accelerator")?,
            saved_gpu_device: row.get("saved_gpu_device")?,
            recommended_backend: row.get("recommended_backend")?,
            recommended_device: row.get("recommended_device")?,
            cleanup_mode: row.get("cleanup_mode")?,
            outcome: row.get("outcome")?,
            transcription_total_ms: row.get("transcription_total_ms")?,
            cleanup_total_ms: row.get("cleanup_total_ms")?,
        })
    }

    pub fn recordings_dir(&self) -> &std::path::Path {
        &self.recordings_dir
    }

    /// Save a new history entry to the database.
    /// The WAV file should already have been written to the recordings directory.
    pub fn save_entry(
        &self,
        file_name: String,
        transcription_text: String,
        post_process_requested: bool,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        cleanup_prompt_id: Option<String>,
        cleanup_model_id: Option<String>,
        model_id: Option<String>,
        engine_type: Option<String>,
        language: Option<String>,
        insertion_mode: Option<String>,
        backend: Option<String>,
        device: Option<String>,
        recovery_reason: Option<String>,
        saved_accelerator: Option<String>,
        saved_gpu_device: Option<String>,
        recommended_backend: Option<String>,
        recommended_device: Option<String>,
        cleanup_mode: Option<String>,
        outcome: Option<String>,
        transcription_total_ms: Option<i64>,
        cleanup_total_ms: Option<i64>,
        duration_ms: i64,
    ) -> Result<HistoryEntry> {
        let timestamp = Utc::now().timestamp();
        let title = self.format_timestamp_title(timestamp);

        let conn = self.get_connection()?;
        conn.execute(
            "INSERT INTO transcription_history (
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                post_process_requested,
                model_id,
                engine_type,
                language,
                insertion_mode,
                backend,
                device,
                saved_accelerator,
                saved_gpu_device,
                recommended_backend,
                recommended_device,
                cleanup_mode,
                outcome,
                transcription_total_ms,
                cleanup_total_ms,
                duration_ms,
                recovery_reason
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            params![
                &file_name,
                timestamp,
                false,
                &title,
                &transcription_text,
                &post_processed_text,
                &post_process_prompt,
                &cleanup_prompt_id,
                &cleanup_model_id,
                post_process_requested,
                &model_id,
                &engine_type,
                &language,
                &insertion_mode,
                &backend,
                &device,
                &saved_accelerator,
                &saved_gpu_device,
                &recommended_backend,
                &recommended_device,
                &cleanup_mode,
                &outcome,
                transcription_total_ms,
                cleanup_total_ms,
                duration_ms,
                &recovery_reason,
            ],
        )?;

        let entry = HistoryEntry {
            id: conn.last_insert_rowid(),
            file_name,
            timestamp,
            duration_ms: Some(duration_ms),
            saved: false,
            title,
            transcription_text,
            post_processed_text,
            post_process_prompt,
            cleanup_prompt_id,
            cleanup_model_id,
            post_process_requested,
            model_id,
            engine_type,
            language,
            insertion_mode,
            backend,
            device,
            recovery_reason,
            saved_accelerator,
            saved_gpu_device,
            recommended_backend,
            recommended_device,
            cleanup_mode,
            outcome,
            transcription_total_ms,
            cleanup_total_ms,
        };

        debug!("Saved history entry with id {}", entry.id);

        // The row is already durable at this point. Retention maintenance must
        // not turn a successful insert into a reported history-save failure.
        if let Err(error) = self.cleanup_old_entries() {
            error!("History audio cleanup after save failed: {error}");
        }

        // Emit typed event for real-time frontend updates
        if let Err(e) = (HistoryUpdatePayload::Added {
            entry: entry.clone(),
        })
        .emit(&self.app_handle)
        {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(entry)
    }

    /// Update an existing history entry with new transcription results (used by retry).
    pub fn update_transcription(
        &self,
        id: i64,
        transcription_text: String,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        model_id: Option<String>,
        engine_type: Option<String>,
        language: Option<String>,
        backend: Option<String>,
        device: Option<String>,
        recovery_reason: Option<String>,
        saved_accelerator: Option<String>,
        saved_gpu_device: Option<String>,
        recommended_backend: Option<String>,
        recommended_device: Option<String>,
        cleanup_mode: Option<String>,
        transcription_total_ms: Option<i64>,
        cleanup_total_ms: Option<i64>,
    ) -> Result<HistoryEntry> {
        let conn = self.get_connection()?;
        let updated = conn.execute(
            "UPDATE transcription_history
             SET transcription_text = ?1,
                 post_processed_text = ?2,
                 post_process_prompt = ?3,
                 cleanup_prompt_id = NULL,
                 cleanup_model_id = NULL,
                 model_id = ?4,
                 engine_type = ?5,
                 language = ?6,
                 backend = ?7,
                 device = ?8,
                 saved_accelerator = ?9,
                 saved_gpu_device = ?10,
                 recommended_backend = ?11,
                 recommended_device = ?12,
                 cleanup_mode = ?13,
                 transcription_total_ms = ?14,
                 cleanup_total_ms = ?15,
                 recovery_reason = ?16,
                 outcome = 'success'
             WHERE id = ?17",
            params![
                transcription_text,
                post_processed_text,
                post_process_prompt,
                model_id,
                engine_type,
                language,
                backend,
                device,
                saved_accelerator,
                saved_gpu_device,
                recommended_backend,
                recommended_device,
                cleanup_mode,
                transcription_total_ms,
                cleanup_total_ms,
                recovery_reason,
                id
            ],
        )?;

        if updated == 0 {
            return Err(anyhow!("History entry {} not found", id));
        }

        let entry = conn
            .query_row(
                "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, cleanup_prompt_id, cleanup_model_id, post_process_requested, model_id, engine_type, duration_ms, language, insertion_mode, backend, device, saved_accelerator, saved_gpu_device, recommended_backend, recommended_device, cleanup_mode, outcome, transcription_total_ms, cleanup_total_ms, recovery_reason
                 FROM transcription_history WHERE id = ?1",
                params![id],
                Self::map_history_entry,
            )?;

        debug!("Updated transcription for history entry {}", id);

        if let Err(e) = (HistoryUpdatePayload::Updated {
            entry: entry.clone(),
        })
        .emit(&self.app_handle)
        {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(entry)
    }

    pub fn update_cleanup(
        &self,
        id: i64,
        post_processed_text: String,
        post_process_prompt: Option<String>,
        cleanup_prompt_id: Option<String>,
        cleanup_model_id: Option<String>,
        cleanup_mode: Option<String>,
        cleanup_total_ms: Option<i64>,
    ) -> Result<HistoryEntry> {
        let conn = self.get_connection()?;
        let entry = Self::update_cleanup_with_conn(
            &conn,
            id,
            post_processed_text,
            post_process_prompt,
            cleanup_prompt_id,
            cleanup_model_id,
            cleanup_mode,
            cleanup_total_ms,
        )?;

        debug!("Updated cleanup for history entry {}", id);

        if let Err(e) = (HistoryUpdatePayload::Updated {
            entry: entry.clone(),
        })
        .emit(&self.app_handle)
        {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(entry)
    }

    fn update_cleanup_with_conn(
        conn: &Connection,
        id: i64,
        post_processed_text: String,
        post_process_prompt: Option<String>,
        cleanup_prompt_id: Option<String>,
        cleanup_model_id: Option<String>,
        cleanup_mode: Option<String>,
        cleanup_total_ms: Option<i64>,
    ) -> Result<HistoryEntry> {
        let updated = conn.execute(
            "UPDATE transcription_history
             SET post_processed_text = ?1,
                 post_process_prompt = ?2,
                 cleanup_prompt_id = ?3,
                 cleanup_model_id = ?4,
                 post_process_requested = 1,
                 cleanup_mode = ?5,
                 cleanup_total_ms = ?6
             WHERE id = ?7",
            params![
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                cleanup_mode,
                cleanup_total_ms,
                id
            ],
        )?;

        if updated == 0 {
            return Err(anyhow!("History entry {} not found", id));
        }

        conn.query_row(
            "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, cleanup_prompt_id, cleanup_model_id, post_process_requested, model_id, engine_type, duration_ms, language, insertion_mode, backend, device, saved_accelerator, saved_gpu_device, recommended_backend, recommended_device, cleanup_mode, outcome, transcription_total_ms, cleanup_total_ms, recovery_reason
             FROM transcription_history WHERE id = ?1",
            params![id],
            Self::map_history_entry,
        )
        .map_err(Into::into)
    }

    pub fn cleanup_old_entries(&self) -> Result<()> {
        let retention_period = crate::settings::get_recording_retention_period(&self.app_handle);
        let conn = self.get_connection()?;
        let now = Utc::now().timestamp();

        // An audio file without a history row cannot be reached from the UI or
        // retried. Clean stale managed orphans regardless of the selected
        // retention policy, while leaving very recent files alone so an
        // in-flight WAV save cannot race a settings change.
        let orphan_count = Self::cleanup_orphan_audio_with_conn(&conn, &self.recordings_dir, now)?;
        if orphan_count > 0 {
            debug!("Cleaned up {} orphaned recording files", orphan_count);
        }

        let deleted_count = match retention_period {
            crate::settings::RecordingRetentionPeriod::Never => 0,
            crate::settings::RecordingRetentionPeriod::PreserveLimit => {
                let limit = crate::settings::get_history_limit(&self.app_handle);
                Self::cleanup_audio_by_count_with_conn(&conn, &self.recordings_dir, limit)?
            }
            _ => Self::cleanup_audio_by_time_with_conn(
                &conn,
                &self.recordings_dir,
                retention_period,
                now,
            )?,
        };

        if deleted_count > 0 {
            debug!(
                "Cleaned up {} retained recording files according to {:?}",
                deleted_count, retention_period
            );
        }

        Ok(())
    }

    /// Resolve a history-owned recording name without permitting a database
    /// value to escape the recordings directory.
    fn retained_audio_path(recordings_dir: &Path, file_name: &str) -> Result<PathBuf> {
        let mut components = Path::new(file_name).components();
        match (components.next(), components.next()) {
            (Some(Component::Normal(_)), None) => Ok(recordings_dir.join(file_name)),
            _ => Err(anyhow!("Invalid retained audio file name: {file_name}")),
        }
    }

    /// Remove one retained recording. A missing file is already in the desired
    /// state; other filesystem failures are surfaced to callers that need an
    /// all-or-nothing history deletion.
    fn remove_retained_audio(recordings_dir: &Path, file_name: &str) -> Result<bool> {
        let file_path = Self::retained_audio_path(recordings_dir, file_name)?;
        match fs::remove_file(&file_path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(anyhow!(
                "Failed to delete retained audio {file_name}: {error}"
            )),
        }
    }

    /// Best-effort policy cleanup deliberately removes only WAV files. The
    /// transcription_history table remains the single transcript store, so an
    /// audio retention policy can expire recordings without erasing searchable
    /// history or creating a shadow transcript database.
    fn remove_audio_candidates(recordings_dir: &Path, file_names: Vec<String>) -> usize {
        let mut deleted_count = 0;
        let mut seen = HashSet::new();
        for file_name in file_names {
            if !seen.insert(file_name.clone()) {
                continue;
            }
            match Self::remove_retained_audio(recordings_dir, &file_name) {
                Ok(true) => {
                    debug!("Deleted retained WAV file: {}", file_name);
                    deleted_count += 1;
                }
                Ok(false) => {}
                Err(error) => {
                    // Retention cleanup is fail-safe: a filesystem error must
                    // never delete or corrupt the history row. A later cleanup
                    // pass can retry the same path.
                    error!("{error}");
                }
            }
        }
        deleted_count
    }

    fn cleanup_audio_by_count_with_conn(
        conn: &Connection,
        recordings_dir: &Path,
        limit: usize,
    ) -> Result<usize> {
        // `saved` protects the history row, not an unlimited WAV. Counting every
        // row makes the selected recording limit an actual upper bound.
        let mut stmt = conn.prepare(
            "SELECT file_name FROM transcription_history ORDER BY timestamp DESC, id DESC",
        )?;
        let file_names = stmt
            .query_map([], |row| row.get::<_, String>("file_name"))?
            .skip(limit)
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self::remove_audio_candidates(recordings_dir, file_names))
    }

    fn cleanup_audio_by_time_with_conn(
        conn: &Connection,
        recordings_dir: &Path,
        retention_period: crate::settings::RecordingRetentionPeriod,
        now: i64,
    ) -> Result<usize> {
        let cutoff_timestamp = match retention_period {
            crate::settings::RecordingRetentionPeriod::Days3 => now - (3 * 24 * 60 * 60),
            crate::settings::RecordingRetentionPeriod::Weeks2 => now - (2 * 7 * 24 * 60 * 60),
            crate::settings::RecordingRetentionPeriod::Months3 => now - (3 * 30 * 24 * 60 * 60),
            _ => unreachable!("time cleanup requires a time-based retention period"),
        };

        let mut stmt = conn.prepare(
            "SELECT file_name FROM transcription_history WHERE timestamp < ?1 ORDER BY timestamp ASC, id ASC",
        )?;
        let file_names = stmt
            .query_map(params![cutoff_timestamp], |row| {
                row.get::<_, String>("file_name")
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self::remove_audio_candidates(recordings_dir, file_names))
    }

    const ORPHAN_AUDIO_GRACE_SECONDS: i64 = 5 * 60;

    fn managed_recording_timestamp(file_name: &str) -> Option<i64> {
        let timestamp = file_name.strip_prefix("handy-")?.strip_suffix(".wav")?;
        if timestamp.is_empty() || !timestamp.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        timestamp.parse().ok()
    }

    fn cleanup_orphan_audio_with_conn(
        conn: &Connection,
        recordings_dir: &Path,
        now: i64,
    ) -> Result<usize> {
        let mut stmt = conn.prepare("SELECT file_name FROM transcription_history")?;
        let referenced = stmt
            .query_map([], |row| row.get::<_, String>("file_name"))?
            .collect::<std::result::Result<HashSet<_>, _>>()?;
        let stale_before = now.saturating_sub(Self::ORPHAN_AUDIO_GRACE_SECONDS);
        let mut deleted_count = 0;

        let entries = match fs::read_dir(recordings_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error.into()),
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    error!("Failed to inspect recordings directory entry: {error}");
                    continue;
                }
            };
            if !entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
            {
                continue;
            }
            let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Some(timestamp) = Self::managed_recording_timestamp(&file_name) else {
                continue;
            };
            if timestamp > stale_before || referenced.contains(&file_name) {
                continue;
            }

            match Self::remove_retained_audio(recordings_dir, &file_name) {
                Ok(true) => deleted_count += 1,
                Ok(false) => {}
                Err(error) => error!("{error}"),
            }
        }

        Ok(deleted_count)
    }

    pub async fn get_history_entries(
        &self,
        cursor: Option<i64>,
        limit: Option<usize>,
        search: Option<&str>,
        start_timestamp: Option<i64>,
        end_timestamp_exclusive: Option<i64>,
        model_filter: Option<&str>,
        outcome_filter: Option<&str>,
        cleanup_filter: Option<&str>,
    ) -> Result<PaginatedHistory> {
        let conn = self.get_connection()?;
        Self::get_history_entries_filtered_with_conn(
            &conn,
            cursor,
            limit,
            search,
            start_timestamp,
            end_timestamp_exclusive,
            model_filter,
            outcome_filter,
            cleanup_filter,
        )
    }

    fn get_history_entries_with_conn(
        conn: &Connection,
        cursor: Option<i64>,
        limit: Option<usize>,
        search: Option<&str>,
        start_timestamp: Option<i64>,
        end_timestamp_exclusive: Option<i64>,
        model_filter: Option<&str>,
        outcome_filter: Option<&str>,
    ) -> Result<PaginatedHistory> {
        Self::get_history_entries_filtered_with_conn(
            conn,
            cursor,
            limit,
            search,
            start_timestamp,
            end_timestamp_exclusive,
            model_filter,
            outcome_filter,
            None,
        )
    }

    fn get_history_entries_filtered_with_conn(
        conn: &Connection,
        cursor: Option<i64>,
        limit: Option<usize>,
        search: Option<&str>,
        start_timestamp: Option<i64>,
        end_timestamp_exclusive: Option<i64>,
        model_filter: Option<&str>,
        outcome_filter: Option<&str>,
        cleanup_filter: Option<&str>,
    ) -> Result<PaginatedHistory> {
        let limit = limit.map(|l| l.min(100));
        let fetch_count = limit
            .map(|lim| lim.saturating_add(1) as i64)
            .unwrap_or(i64::MAX);
        let search_pattern = Self::history_search_pattern(search);
        let outcome_filter = match outcome_filter {
            None => None,
            Some("success") => Some("success"),
            Some("failure") => Some("failure"),
            Some(value) => return Err(anyhow!("Invalid history outcome filter: {value}")),
        };
        let cleanup_filter = match cleanup_filter {
            None => None,
            Some("requested") => Some(true),
            Some("not_requested") => Some(false),
            Some(value) => return Err(anyhow!("Invalid history cleanup filter: {value}")),
        };

        let mut stmt = conn.prepare(
            "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, cleanup_prompt_id, cleanup_model_id, post_process_requested, model_id, engine_type, duration_ms, language, insertion_mode, backend, device, saved_accelerator, saved_gpu_device, recommended_backend, recommended_device, cleanup_mode, outcome, transcription_total_ms, cleanup_total_ms, recovery_reason
             FROM transcription_history
             WHERE (?1 IS NULL OR id < ?1)
               AND (
                    ?2 IS NULL
                    OR transcription_text LIKE ?2 ESCAPE '\\'
                    OR COALESCE(post_processed_text, '') LIKE ?2 ESCAPE '\\'
               )
               AND (?3 IS NULL OR timestamp >= ?3)
               AND (?4 IS NULL OR timestamp < ?4)
               AND (?5 IS NULL OR model_id = ?5)
               AND (
                    ?6 IS NULL
                    OR COALESCE(
                        outcome,
                        CASE WHEN transcription_text != '' THEN 'success' ELSE 'failure' END
                    ) = ?6
               )
               AND (?7 IS NULL OR post_process_requested = ?7)
             ORDER BY id DESC
             LIMIT ?8",
        )?;
        let mut entries = stmt
            .query_map(
                params![
                    cursor,
                    search_pattern,
                    start_timestamp,
                    end_timestamp_exclusive,
                    model_filter,
                    outcome_filter,
                    cleanup_filter,
                    fetch_count
                ],
                Self::map_history_entry,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let has_more = limit.is_some_and(|lim| entries.len() > lim);
        if has_more {
            entries.pop();
        }

        Ok(PaginatedHistory { entries, has_more })
    }

    fn history_search_pattern(search: Option<&str>) -> Option<String> {
        let search = search?.trim();
        if search.is_empty() {
            return None;
        }

        let mut escaped = String::with_capacity(search.len());
        for ch in search.chars() {
            if matches!(ch, '\\' | '%' | '_') {
                escaped.push('\\');
            }
            escaped.push(ch);
        }
        Some(format!("%{escaped}%"))
    }

    #[cfg(test)]
    fn get_latest_entry_with_conn(conn: &Connection) -> Result<Option<HistoryEntry>> {
        let mut stmt = conn.prepare(
            "SELECT
                id,
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                post_process_requested,
                model_id,
                engine_type,
                duration_ms,
                language,
                insertion_mode,
                backend,
                device,
                saved_accelerator,
                saved_gpu_device,
                recommended_backend,
                recommended_device,
                cleanup_mode,
                outcome,
                transcription_total_ms,
                cleanup_total_ms,
                recovery_reason
             FROM transcription_history
             ORDER BY timestamp DESC
             LIMIT 1",
        )?;

        let entry = stmt.query_row([], Self::map_history_entry).optional()?;
        Ok(entry)
    }

    /// Get the latest entry with non-empty transcription text.
    pub fn get_latest_completed_entry(&self) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        Self::get_latest_completed_entry_with_conn(&conn)
    }

    fn get_latest_completed_entry_with_conn(conn: &Connection) -> Result<Option<HistoryEntry>> {
        let mut stmt = conn.prepare(
            "SELECT
                id,
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                post_process_requested,
                model_id,
                engine_type,
                duration_ms,
                language,
                insertion_mode,
                backend,
                device,
                saved_accelerator,
                saved_gpu_device,
                recommended_backend,
                recommended_device,
                cleanup_mode,
                outcome,
                transcription_total_ms,
                cleanup_total_ms,
                recovery_reason
             FROM transcription_history
             WHERE transcription_text != ''
             ORDER BY timestamp DESC
             LIMIT 1",
        )?;

        let entry = stmt.query_row([], Self::map_history_entry).optional()?;
        Ok(entry)
    }

    pub async fn toggle_saved_status(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;

        // Get current saved status
        let current_saved: bool = conn.query_row(
            "SELECT saved FROM transcription_history WHERE id = ?1",
            params![id],
            |row| row.get("saved"),
        )?;

        let new_saved = !current_saved;

        conn.execute(
            "UPDATE transcription_history SET saved = ?1 WHERE id = ?2",
            params![new_saved, id],
        )?;

        debug!("Toggled saved status for entry {}: {}", id, new_saved);

        // Emit history updated event
        if let Err(e) = (HistoryUpdatePayload::Toggled { id }).emit(&self.app_handle) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    pub fn get_audio_file_path(&self, file_name: &str) -> Result<PathBuf> {
        let path = Self::retained_audio_path(&self.recordings_dir, file_name)?;
        if path.is_file() {
            Ok(path)
        } else {
            Err(anyhow!("Recording audio is no longer retained"))
        }
    }

    pub async fn get_entry_by_id(&self, id: i64) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT
                id,
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                post_process_requested,
                model_id,
                engine_type,
                duration_ms,
                language,
                insertion_mode,
                backend,
                device,
                saved_accelerator,
                saved_gpu_device,
                recommended_backend,
                recommended_device,
                cleanup_mode,
                outcome,
                transcription_total_ms,
                cleanup_total_ms,
                recovery_reason
             FROM transcription_history
             WHERE id = ?1",
        )?;

        let entry = stmt.query_row([id], Self::map_history_entry).optional()?;

        Ok(entry)
    }

    fn delete_entry_with_conn(conn: &Connection, recordings_dir: &Path, id: i64) -> Result<bool> {
        let file_name = conn
            .query_row(
                "SELECT file_name FROM transcription_history WHERE id = ?1",
                params![id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(file_name) = file_name else {
            return Ok(false);
        };

        // A generated recording name should be unique, but preserve the file if
        // a legacy/corrupt database has another row pointing at the same WAV.
        let other_references: i64 = conn.query_row(
            "SELECT COUNT(*) FROM transcription_history WHERE file_name = ?1 AND id != ?2",
            params![&file_name, id],
            |row| row.get(0),
        )?;
        if other_references == 0 {
            // Do not drop the only row that identifies a retained file when the
            // filesystem refuses deletion. Keeping the row makes the failure
            // visible/retryable instead of silently creating an orphan.
            Self::remove_retained_audio(recordings_dir, &file_name)?;
        }

        conn.execute(
            "DELETE FROM transcription_history WHERE id = ?1",
            params![id],
        )?;
        Ok(true)
    }

    pub async fn delete_entry(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;
        let deleted = Self::delete_entry_with_conn(&conn, &self.recordings_dir, id)?;

        if deleted {
            debug!("Deleted history entry with id: {}", id);
        }

        // Preserve the existing idempotent command contract: deleting a missing
        // row is a no-op, but the frontend may still discard its stale copy.
        if let Err(e) = (HistoryUpdatePayload::Deleted { id }).emit(&self.app_handle) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    fn format_timestamp_title(&self, timestamp: i64) -> String {
        if let Some(utc_datetime) = DateTime::from_timestamp(timestamp, 0) {
            // Convert UTC to local timezone
            let local_datetime = utc_datetime.with_timezone(&Local);
            local_datetime.format("%B %e, %Y - %l:%M%p").to_string()
        } else {
            format!("Recording {}", timestamp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};
    use tempfile::TempDir;

    fn setup_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch(
            "CREATE TABLE transcription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                saved BOOLEAN NOT NULL DEFAULT 0,
                title TEXT NOT NULL,
                transcription_text TEXT NOT NULL,
                post_processed_text TEXT,
                post_process_prompt TEXT,
                cleanup_prompt_id TEXT,
                cleanup_model_id TEXT,
                post_process_requested BOOLEAN NOT NULL DEFAULT 0,
                model_id TEXT,
                engine_type TEXT,
                duration_ms INTEGER,
                language TEXT,
                insertion_mode TEXT,
                backend TEXT,
                device TEXT,
                saved_accelerator TEXT,
                saved_gpu_device TEXT,
                recommended_backend TEXT,
                recommended_device TEXT,
                cleanup_mode TEXT,
                outcome TEXT,
                transcription_total_ms INTEGER,
                cleanup_total_ms INTEGER,
                recovery_reason TEXT
            );",
        )
        .expect("create transcription_history table");
        conn
    }

    fn insert_entry(conn: &Connection, timestamp: i64, text: &str, post_processed: Option<&str>) {
        insert_entry_with_model(conn, timestamp, text, post_processed, None);
    }

    fn insert_entry_with_model(
        conn: &Connection,
        timestamp: i64,
        text: &str,
        post_processed: Option<&str>,
        model_id: Option<&str>,
    ) {
        conn.execute(
            "INSERT INTO transcription_history (
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                cleanup_prompt_id,
                cleanup_model_id,
                post_process_requested,
                model_id,
                language,
                duration_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                format!("handy-{}.wav", timestamp),
                timestamp,
                false,
                format!("Recording {}", timestamp),
                text,
                post_processed,
                Option::<String>::None,
                Option::<String>::None,
                Option::<String>::None,
                false,
                model_id,
                Option::<String>::None,
                Option::<i64>::None,
            ],
        )
        .expect("insert history entry");
    }

    fn write_recording(dir: &TempDir, timestamp: i64) -> PathBuf {
        let path = dir.path().join(format!("handy-{timestamp}.wav"));
        fs::write(&path, b"wav").expect("write recording fixture");
        path
    }

    fn row_count(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM transcription_history", [], |row| {
            row.get(0)
        })
        .expect("count history rows")
    }

    #[test]
    fn migrations_add_model_id_without_losing_existing_history() {
        let mut conn = Connection::open_in_memory().expect("open in-memory db");
        Migrations::new(MIGRATIONS[..4].to_vec())
            .to_latest(&mut conn)
            .expect("apply pre-model history migrations");
        conn.execute(
            "INSERT INTO transcription_history (
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                post_process_requested
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                "handy-old.wav",
                100_i64,
                false,
                "Old recording",
                "legacy raw text",
                Some("legacy final text"),
                Option::<String>::None,
                true,
            ],
        )
        .expect("insert pre-model history row");

        Migrations::new(MIGRATIONS.to_vec())
            .to_latest(&mut conn)
            .expect("apply model history migration");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("load migrated history")
            .expect("migrated entry exists");
        assert_eq!(entry.transcription_text, "legacy raw text");
        assert_eq!(
            entry.post_processed_text.as_deref(),
            Some("legacy final text")
        );
        assert!(entry.post_process_requested);
        assert!(entry.model_id.is_none());
        assert!(entry.engine_type.is_none());
        assert!(entry.duration_ms.is_none());
        assert!(entry.language.is_none());
        assert!(entry.insertion_mode.is_none());
        assert!(entry.backend.is_none());
        assert!(entry.device.is_none());
        assert!(entry.recovery_reason.is_none());
        assert!(entry.saved_accelerator.is_none());
        assert!(entry.saved_gpu_device.is_none());
        assert!(entry.recommended_backend.is_none());
        assert!(entry.recommended_device.is_none());
        assert!(entry.cleanup_prompt_id.is_none());
        assert!(entry.cleanup_model_id.is_none());
        assert!(entry.cleanup_mode.is_none());
        assert!(entry.outcome.is_none());
        assert!(entry.transcription_total_ms.is_none());
        assert!(entry.cleanup_total_ms.is_none());
    }

    #[test]
    fn history_entry_preserves_recording_duration() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "timed recording", None);
        conn.execute(
            "UPDATE transcription_history SET duration_ms = ?1 WHERE timestamp = ?2",
            params![1_234_i64, 100_i64],
        )
        .expect("store recording duration");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch timed entry")
            .expect("timed entry exists");
        assert_eq!(entry.duration_ms, Some(1_234));
    }

    #[test]
    fn history_entry_preserves_engine_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "engine-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET engine_type = ?1 WHERE timestamp = ?2",
            params!["transcribe_cpp", 100_i64],
        )
        .expect("store engine metadata");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch engine-tagged entry")
            .expect("engine-tagged entry exists");
        assert_eq!(entry.engine_type.as_deref(), Some("transcribe_cpp"));
    }

    #[test]
    fn history_entry_preserves_insertion_mode_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "insertion-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET insertion_mode = ?1 WHERE timestamp = ?2",
            params!["at_stop", 100_i64],
        )
        .expect("store insertion mode metadata");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch insertion-tagged entry")
            .expect("insertion-tagged entry exists");
        assert_eq!(entry.insertion_mode.as_deref(), Some("at_stop"));
    }

    #[test]
    fn history_entry_preserves_saved_recommended_and_runtime_compute_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "compute-plan-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET saved_accelerator = ?1, saved_gpu_device = ?2, recommended_backend = ?3, recommended_device = ?4, backend = ?5, device = ?6, recovery_reason = ?7 WHERE timestamp = ?8",
            params!["gpu", "stable-device-id", "auto", "Discrete GPU", "vulkan", "gpu-0", "runtime_health_failure", 100_i64],
        )
        .expect("store compute-plan metadata");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch compute-plan-tagged entry")
            .expect("compute-plan-tagged entry exists");
        assert_eq!(entry.saved_accelerator.as_deref(), Some("gpu"));
        assert_eq!(entry.saved_gpu_device.as_deref(), Some("stable-device-id"));
        assert_eq!(entry.recommended_backend.as_deref(), Some("auto"));
        assert_eq!(entry.recommended_device.as_deref(), Some("Discrete GPU"));
        assert_eq!(entry.backend.as_deref(), Some("vulkan"));
        assert_eq!(entry.device.as_deref(), Some("gpu-0"));
        assert_eq!(
            entry.recovery_reason.as_deref(),
            Some("runtime_health_failure")
        );
    }

    #[test]
    fn history_entry_preserves_cleanup_mode_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "cleanup-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET cleanup_mode = ?1 WHERE timestamp = ?2",
            params!["provider:openai", 100_i64],
        )
        .expect("store cleanup mode metadata");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch cleanup-tagged entry")
            .expect("cleanup-tagged entry exists");
        assert_eq!(entry.cleanup_mode.as_deref(), Some("provider:openai"));
    }

    #[test]
    fn history_entry_preserves_outcome_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "outcome-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET outcome = ?1 WHERE timestamp = ?2",
            params!["success", 100_i64],
        )
        .expect("store outcome metadata");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch outcome-tagged entry")
            .expect("outcome-tagged entry exists");
        assert_eq!(entry.outcome.as_deref(), Some("success"));
    }

    #[test]
    fn history_entry_preserves_stage_timing_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "timed stages", None);
        conn.execute(
            "UPDATE transcription_history SET transcription_total_ms = ?1, cleanup_total_ms = ?2 WHERE timestamp = ?3",
            params![321_i64, 87_i64, 100_i64],
        )
        .expect("store history stage timings");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch stage-timed entry")
            .expect("stage-timed entry exists");
        assert_eq!(entry.transcription_total_ms, Some(321));
        assert_eq!(entry.cleanup_total_ms, Some(87));
    }

    #[test]
    fn history_entry_preserves_language_metadata() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "language-tagged recording", None);
        conn.execute(
            "UPDATE transcription_history SET language = ?1 WHERE timestamp = ?2",
            params!["en", 100_i64],
        )
        .expect("store recording language");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch language-tagged entry")
            .expect("language-tagged entry exists");
        assert_eq!(entry.language.as_deref(), Some("en"));
    }

    #[test]
    fn get_latest_entry_returns_none_when_empty() {
        let conn = setup_conn();
        let entry = HistoryManager::get_latest_entry_with_conn(&conn).expect("fetch latest entry");
        assert!(entry.is_none());
    }

    #[test]
    fn get_latest_entry_returns_newest_entry() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "first", None);
        insert_entry(&conn, 200, "second", Some("processed"));

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch latest entry")
            .expect("entry exists");

        assert_eq!(entry.timestamp, 200);
        assert_eq!(entry.transcription_text, "second");
        assert_eq!(entry.post_processed_text.as_deref(), Some("processed"));
    }

    #[test]
    fn get_latest_completed_entry_skips_empty_entries() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "completed", None);
        insert_entry(&conn, 200, "", None);

        let entry = HistoryManager::get_latest_completed_entry_with_conn(&conn)
            .expect("fetch latest completed entry")
            .expect("completed entry exists");

        assert_eq!(entry.timestamp, 100);
        assert_eq!(entry.transcription_text, "completed");
    }

    #[test]
    fn retry_cleanup_updates_only_cleanup_fields() {
        let conn = setup_conn();
        insert_entry_with_model(
            &conn,
            100,
            "raw transcript",
            Some("old cleanup"),
            Some("whisper-large-v3-turbo"),
        );

        let entry = HistoryManager::update_cleanup_with_conn(
            &conn,
            1,
            "new cleanup".to_string(),
            Some("new prompt".to_string()),
            Some("prompt-id".to_string()),
            Some("cleanup-model".to_string()),
            Some("provider:openai".to_string()),
            Some(42),
        )
        .expect("update cleanup without retranscribing");

        assert_eq!(entry.transcription_text, "raw transcript");
        assert_eq!(entry.post_processed_text.as_deref(), Some("new cleanup"));
        assert_eq!(entry.post_process_prompt.as_deref(), Some("new prompt"));
        assert_eq!(entry.cleanup_prompt_id.as_deref(), Some("prompt-id"));
        assert_eq!(entry.cleanup_model_id.as_deref(), Some("cleanup-model"));
        assert!(entry.post_process_requested);
        assert_eq!(entry.cleanup_mode.as_deref(), Some("provider:openai"));
        assert_eq!(entry.cleanup_total_ms, Some(42));
        assert_eq!(entry.model_id.as_deref(), Some("whisper-large-v3-turbo"));
    }

    #[test]
    fn retry_cleanup_rejects_missing_history_entry() {
        let conn = setup_conn();
        let error = HistoryManager::update_cleanup_with_conn(
            &conn,
            99,
            "new cleanup".to_string(),
            None,
            None,
            None,
            Some("provider:openai".to_string()),
            Some(10),
        )
        .expect_err("missing history entry should fail");

        assert!(error.to_string().contains("History entry 99 not found"));
    }

    #[test]
    fn history_search_matches_raw_and_final_text_case_insensitively() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "Alpha raw phrase", None);
        insert_entry(&conn, 200, "different raw", Some("Polished Beta phrase"));
        insert_entry(&conn, 300, "unrelated", None);

        let raw = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            Some("alpha"),
            None,
            None,
            None,
            None,
        )
        .expect("search raw text");
        assert_eq!(raw.entries.len(), 1);
        assert_eq!(raw.entries[0].timestamp, 100);

        let final_text = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            Some("BETA"),
            None,
            None,
            None,
            None,
        )
        .expect("search final text");
        assert_eq!(final_text.entries.len(), 1);
        assert_eq!(final_text.entries[0].timestamp, 200);
    }

    #[test]
    fn history_search_preserves_literal_wildcards_and_cursor_pagination() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "needle 100% first", None);
        insert_entry(&conn, 200, "needle 100% second", None);
        insert_entry(&conn, 300, "needle 100 percent unrelated", None);

        let first_page = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(1),
            Some("100%"),
            None,
            None,
            None,
            None,
        )
        .expect("first search page");
        assert_eq!(first_page.entries.len(), 1);
        assert_eq!(first_page.entries[0].timestamp, 200);
        assert!(first_page.has_more);

        let second_page = HistoryManager::get_history_entries_with_conn(
            &conn,
            Some(first_page.entries[0].id),
            Some(1),
            Some("100%"),
            None,
            None,
            None,
            None,
        )
        .expect("second search page");
        assert_eq!(second_page.entries.len(), 1);
        assert_eq!(second_page.entries[0].timestamp, 100);
        assert!(!second_page.has_more);
    }

    #[test]
    fn history_date_filter_uses_inclusive_start_and_exclusive_end() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "before", None);
        insert_entry(&conn, 200, "inside", None);
        insert_entry(&conn, 300, "at end", None);

        let filtered = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            Some(200),
            Some(300),
            None,
            None,
        )
        .expect("filter history by date range");

        assert_eq!(filtered.entries.len(), 1);
        assert_eq!(filtered.entries[0].timestamp, 200);
    }

    #[test]
    fn history_date_filter_combines_with_search() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "target before range", None);
        insert_entry(&conn, 200, "target in range", None);
        insert_entry(&conn, 250, "different in range", None);

        let filtered = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            Some("target"),
            Some(150),
            Some(250),
            None,
            None,
        )
        .expect("combine search and date range");

        assert_eq!(filtered.entries.len(), 1);
        assert_eq!(filtered.entries[0].timestamp, 200);
    }

    #[test]
    fn history_model_filter_matches_only_selected_model_and_combines_with_search() {
        let conn = setup_conn();
        insert_entry_with_model(
            &conn,
            100,
            "target whisper",
            None,
            Some("whisper-large-v3-turbo"),
        );
        insert_entry_with_model(
            &conn,
            200,
            "target parakeet",
            None,
            Some("parakeet-tdt-0.6b-v3"),
        );
        insert_entry_with_model(
            &conn,
            300,
            "other parakeet",
            None,
            Some("parakeet-tdt-0.6b-v3"),
        );
        insert_entry(&conn, 400, "target unknown", None);

        let filtered = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            Some("target"),
            None,
            None,
            Some("parakeet-tdt-0.6b-v3"),
            None,
        )
        .expect("filter history by model");

        assert_eq!(filtered.entries.len(), 1);
        assert_eq!(filtered.entries[0].timestamp, 200);
        assert_eq!(
            filtered.entries[0].model_id.as_deref(),
            Some("parakeet-tdt-0.6b-v3")
        );
    }

    #[test]
    fn history_outcome_filter_distinguishes_success_and_failure() {
        let conn = setup_conn();
        insert_entry_with_model(
            &conn,
            100,
            "successful parakeet",
            None,
            Some("parakeet-tdt-0.6b-v3"),
        );
        insert_entry_with_model(&conn, 200, "", None, Some("parakeet-tdt-0.6b-v3"));
        insert_entry_with_model(
            &conn,
            300,
            "successful whisper",
            None,
            Some("whisper-large-v3-turbo"),
        );

        let successful = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            Some("parakeet-tdt-0.6b-v3"),
            Some("success"),
        )
        .expect("filter successful history");
        assert_eq!(successful.entries.len(), 1);
        assert_eq!(successful.entries[0].timestamp, 100);

        let failed = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            Some("parakeet-tdt-0.6b-v3"),
            Some("failure"),
        )
        .expect("filter failed history");
        assert_eq!(failed.entries.len(), 1);
        assert_eq!(failed.entries[0].timestamp, 200);
    }

    #[test]
    fn history_outcome_filter_prefers_persisted_session_result() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "text exists but session failed", None);
        conn.execute(
            "UPDATE transcription_history SET outcome = 'failure' WHERE timestamp = ?1",
            params![100_i64],
        )
        .expect("persist explicit failure outcome");

        let failed = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            Some("failure"),
        )
        .expect("filter by persisted failure outcome");
        assert_eq!(failed.entries.len(), 1);
        assert_eq!(failed.entries[0].outcome.as_deref(), Some("failure"));

        let successful = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            Some("success"),
        )
        .expect("filter by persisted success outcome");
        assert!(successful.entries.is_empty());
    }

    #[test]
    fn history_cleanup_filter_uses_requested_state_and_combines_with_outcome() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "successful cleaned", Some("cleaned text"));
        insert_entry(&conn, 200, "successful raw", None);
        insert_entry(&conn, 300, "", None);
        conn.execute(
            "UPDATE transcription_history SET post_process_requested = 1 WHERE timestamp IN (?1, ?2)",
            params![100_i64, 300_i64],
        )
        .expect("mark cleanup requested entries");

        let requested_success = HistoryManager::get_history_entries_filtered_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            Some("success"),
            Some("requested"),
        )
        .expect("filter cleanup-requested successful history");
        assert_eq!(requested_success.entries.len(), 1);
        assert_eq!(requested_success.entries[0].timestamp, 100);

        let not_requested = HistoryManager::get_history_entries_filtered_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            Some("success"),
            Some("not_requested"),
        )
        .expect("filter history without cleanup requested");
        assert_eq!(not_requested.entries.len(), 1);
        assert_eq!(not_requested.entries[0].timestamp, 200);
    }

    #[test]
    fn history_cleanup_filter_rejects_unknown_values() {
        let conn = setup_conn();
        let error = HistoryManager::get_history_entries_filtered_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            None,
            Some("unknown"),
        )
        .expect_err("reject unknown history cleanup filter");

        assert!(error.to_string().contains("Invalid history cleanup filter"));
    }

    #[test]
    fn history_outcome_filter_rejects_unknown_values() {
        let conn = setup_conn();
        let error = HistoryManager::get_history_entries_with_conn(
            &conn,
            None,
            Some(10),
            None,
            None,
            None,
            None,
            Some("unknown"),
        )
        .expect_err("reject unknown history outcome filter");

        assert!(error.to_string().contains("Invalid history outcome filter"));
    }

    #[test]
    fn count_retention_prunes_audio_without_pruning_history() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        for timestamp in [100_i64, 200, 300] {
            insert_entry(&conn, timestamp, &format!("transcript {timestamp}"), None);
            write_recording(&recordings, timestamp);
        }
        conn.execute(
            "UPDATE transcription_history SET saved = 1 WHERE timestamp = ?1",
            params![100_i64],
        )
        .expect("star oldest history row");

        let deleted = HistoryManager::cleanup_audio_by_count_with_conn(&conn, recordings.path(), 2)
            .expect("apply count-based audio retention");

        assert_eq!(deleted, 1);
        assert_eq!(
            row_count(&conn),
            3,
            "audio retention must not erase transcripts"
        );
        assert!(!recordings.path().join("handy-100.wav").exists());
        assert!(recordings.path().join("handy-200.wav").exists());
        assert!(recordings.path().join("handy-300.wav").exists());
        let saved: bool = conn
            .query_row(
                "SELECT saved FROM transcription_history WHERE timestamp = ?1",
                params![100_i64],
                |row| row.get(0),
            )
            .expect("saved history row remains");
        assert!(
            saved,
            "starring protects history metadata, not an unbounded WAV"
        );
    }

    #[test]
    fn time_retention_prunes_only_expired_audio_and_keeps_transcripts() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        let day = 24 * 60 * 60;
        let now = 10 * day;
        let old_timestamp = now - (4 * day);
        let recent_timestamp = now - day;
        for timestamp in [old_timestamp, recent_timestamp] {
            insert_entry(&conn, timestamp, &format!("transcript {timestamp}"), None);
            write_recording(&recordings, timestamp);
        }

        let deleted = HistoryManager::cleanup_audio_by_time_with_conn(
            &conn,
            recordings.path(),
            crate::settings::RecordingRetentionPeriod::Days3,
            now,
        )
        .expect("apply time-based audio retention");

        assert_eq!(deleted, 1);
        assert_eq!(row_count(&conn), 2);
        assert!(!recordings
            .path()
            .join(format!("handy-{old_timestamp}.wav"))
            .exists());
        assert!(recordings
            .path()
            .join(format!("handy-{recent_timestamp}.wav"))
            .exists());
    }

    #[test]
    fn deleting_history_removes_its_retained_audio() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        insert_entry(&conn, 100, "delete me", None);
        let audio_path = write_recording(&recordings, 100);

        assert!(
            HistoryManager::delete_entry_with_conn(&conn, recordings.path(), 1)
                .expect("delete history and retained audio")
        );
        assert_eq!(row_count(&conn), 0);
        assert!(!audio_path.exists());
    }

    #[test]
    fn deleting_history_with_already_missing_audio_still_removes_the_row() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        insert_entry(&conn, 100, "audio already expired", None);

        assert!(
            HistoryManager::delete_entry_with_conn(&conn, recordings.path(), 1)
                .expect("missing retained audio is already deleted")
        );
        assert_eq!(row_count(&conn), 0);
    }

    #[test]
    fn history_delete_failure_keeps_row_linked_to_undeleted_audio() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        insert_entry(&conn, 100, "keep on failure", None);
        let audio_path = recordings.path().join("handy-100.wav");
        fs::create_dir(&audio_path).expect("make undeletable-as-file fixture");

        let error = HistoryManager::delete_entry_with_conn(&conn, recordings.path(), 1)
            .expect_err("filesystem deletion failure must stop row deletion");

        assert!(error
            .to_string()
            .contains("Failed to delete retained audio"));
        assert_eq!(
            row_count(&conn),
            1,
            "the row must remain retryable on failure"
        );
        assert!(audio_path.is_dir());
    }

    #[test]
    fn retention_delete_failure_is_best_effort_and_never_removes_history() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        insert_entry(&conn, 100, "old transcript", None);
        insert_entry(&conn, 200, "new transcript", None);
        let blocked_path = recordings.path().join("handy-100.wav");
        fs::create_dir(&blocked_path).expect("make undeletable-as-file fixture");
        write_recording(&recordings, 200);

        let deleted = HistoryManager::cleanup_audio_by_count_with_conn(&conn, recordings.path(), 1)
            .expect("retention cleanup should fail open on per-file errors");

        assert_eq!(deleted, 0);
        assert_eq!(row_count(&conn), 2);
        assert!(blocked_path.is_dir());
    }

    #[test]
    fn orphan_cleanup_removes_only_stale_managed_recordings() {
        let conn = setup_conn();
        let recordings = tempfile::tempdir().expect("create recordings dir");
        insert_entry(&conn, 100, "referenced", None);
        let referenced = write_recording(&recordings, 100);
        let stale_orphan = write_recording(&recordings, 200);
        let recent_orphan = write_recording(&recordings, 950);
        let unmanaged = recordings.path().join("keep-me.wav");
        fs::write(&unmanaged, b"not managed by history").expect("write unmanaged fixture");

        let deleted =
            HistoryManager::cleanup_orphan_audio_with_conn(&conn, recordings.path(), 1_000)
                .expect("clean stale recording orphans");

        assert_eq!(deleted, 1);
        assert!(referenced.exists());
        assert!(!stale_orphan.exists());
        assert!(
            recent_orphan.exists(),
            "recent files get an in-flight grace period"
        );
        assert!(
            unmanaged.exists(),
            "cleanup must not delete unrelated WAV files"
        );
        assert_eq!(row_count(&conn), 1);
    }

    #[test]
    fn invalid_history_audio_path_fails_closed() {
        let recordings = tempfile::tempdir().expect("create recordings dir");
        let outside = recordings
            .path()
            .parent()
            .expect("tempdir parent")
            .join("outside.wav");
        fs::write(&outside, b"outside").expect("write outside fixture");

        let error = HistoryManager::remove_retained_audio(recordings.path(), "../outside.wav")
            .expect_err("path traversal must not be followed");

        assert!(error
            .to_string()
            .contains("Invalid retained audio file name"));
        assert!(outside.exists());
        let _ = fs::remove_file(outside);
    }
}
