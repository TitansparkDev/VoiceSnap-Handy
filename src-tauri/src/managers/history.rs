use anyhow::{anyhow, Result};
use chrono::{DateTime, Local, Utc};
use log::{debug, error, info};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::fs;
use std::path::PathBuf;
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
    pub saved: bool,
    pub title: String,
    pub transcription_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    pub post_process_requested: bool,
    pub model_id: Option<String>,
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

        // Initialize database and run migrations synchronously
        manager.init_database()?;

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
            saved: row.get("saved")?,
            title: row.get("title")?,
            transcription_text: row.get("transcription_text")?,
            post_processed_text: row.get("post_processed_text")?,
            post_process_prompt: row.get("post_process_prompt")?,
            post_process_requested: row.get("post_process_requested")?,
            model_id: row.get("model_id")?,
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
        model_id: Option<String>,
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
                post_process_requested,
                model_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &file_name,
                timestamp,
                false,
                &title,
                &transcription_text,
                &post_processed_text,
                &post_process_prompt,
                post_process_requested,
                &model_id,
            ],
        )?;

        let entry = HistoryEntry {
            id: conn.last_insert_rowid(),
            file_name,
            timestamp,
            saved: false,
            title,
            transcription_text,
            post_processed_text,
            post_process_prompt,
            post_process_requested,
            model_id,
        };

        debug!("Saved history entry with id {}", entry.id);

        self.cleanup_old_entries()?;

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
    ) -> Result<HistoryEntry> {
        let conn = self.get_connection()?;
        let updated = conn.execute(
            "UPDATE transcription_history
             SET transcription_text = ?1,
                 post_processed_text = ?2,
                 post_process_prompt = ?3,
                 model_id = ?4
             WHERE id = ?5",
            params![
                transcription_text,
                post_processed_text,
                post_process_prompt,
                model_id,
                id
            ],
        )?;

        if updated == 0 {
            return Err(anyhow!("History entry {} not found", id));
        }

        let entry = conn
            .query_row(
                "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_requested, model_id
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
    ) -> Result<HistoryEntry> {
        let conn = self.get_connection()?;
        let entry =
            Self::update_cleanup_with_conn(&conn, id, post_processed_text, post_process_prompt)?;

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
    ) -> Result<HistoryEntry> {
        let updated = conn.execute(
            "UPDATE transcription_history
             SET post_processed_text = ?1,
                 post_process_prompt = ?2,
                 post_process_requested = 1
             WHERE id = ?3",
            params![post_processed_text, post_process_prompt, id],
        )?;

        if updated == 0 {
            return Err(anyhow!("History entry {} not found", id));
        }

        conn.query_row(
            "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_requested, model_id
             FROM transcription_history WHERE id = ?1",
            params![id],
            Self::map_history_entry,
        )
        .map_err(Into::into)
    }

    pub fn cleanup_old_entries(&self) -> Result<()> {
        let retention_period = crate::settings::get_recording_retention_period(&self.app_handle);

        match retention_period {
            crate::settings::RecordingRetentionPeriod::Never => {
                // Don't delete anything
                Ok(())
            }
            crate::settings::RecordingRetentionPeriod::PreserveLimit => {
                // Use the old count-based logic with history_limit
                let limit = crate::settings::get_history_limit(&self.app_handle);
                self.cleanup_by_count(limit)
            }
            _ => {
                // Use time-based logic
                self.cleanup_by_time(retention_period)
            }
        }
    }

    fn delete_entries_and_files(&self, entries: &[(i64, String)]) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        let conn = self.get_connection()?;
        let mut deleted_count = 0;

        for (id, file_name) in entries {
            // Delete database entry
            conn.execute(
                "DELETE FROM transcription_history WHERE id = ?1",
                params![id],
            )?;

            // Delete WAV file
            let file_path = self.recordings_dir.join(file_name);
            if file_path.exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to delete WAV file {}: {}", file_name, e);
                } else {
                    debug!("Deleted old WAV file: {}", file_name);
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }

    fn cleanup_by_count(&self, limit: usize) -> Result<()> {
        let conn = self.get_connection()?;

        // Get all entries that are not saved, ordered by timestamp desc
        let mut stmt = conn.prepare(
            "SELECT id, file_name FROM transcription_history WHERE saved = 0 ORDER BY timestamp DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
        })?;

        let mut entries: Vec<(i64, String)> = Vec::new();
        for row in rows {
            entries.push(row?);
        }

        if entries.len() > limit {
            let entries_to_delete = &entries[limit..];
            let deleted_count = self.delete_entries_and_files(entries_to_delete)?;

            if deleted_count > 0 {
                debug!("Cleaned up {} old history entries by count", deleted_count);
            }
        }

        Ok(())
    }

    fn cleanup_by_time(
        &self,
        retention_period: crate::settings::RecordingRetentionPeriod,
    ) -> Result<()> {
        let conn = self.get_connection()?;

        // Calculate cutoff timestamp (current time minus retention period)
        let now = Utc::now().timestamp();
        let cutoff_timestamp = match retention_period {
            crate::settings::RecordingRetentionPeriod::Days3 => now - (3 * 24 * 60 * 60), // 3 days in seconds
            crate::settings::RecordingRetentionPeriod::Weeks2 => now - (2 * 7 * 24 * 60 * 60), // 2 weeks in seconds
            crate::settings::RecordingRetentionPeriod::Months3 => now - (3 * 30 * 24 * 60 * 60), // 3 months in seconds (approximate)
            _ => unreachable!("Should not reach here"),
        };

        // Get all unsaved entries older than the cutoff timestamp
        let mut stmt = conn.prepare(
            "SELECT id, file_name FROM transcription_history WHERE saved = 0 AND timestamp < ?1",
        )?;

        let rows = stmt.query_map(params![cutoff_timestamp], |row| {
            Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
        })?;

        let mut entries_to_delete: Vec<(i64, String)> = Vec::new();
        for row in rows {
            entries_to_delete.push(row?);
        }

        let deleted_count = self.delete_entries_and_files(&entries_to_delete)?;

        if deleted_count > 0 {
            debug!(
                "Cleaned up {} old history entries based on retention period",
                deleted_count
            );
        }

        Ok(())
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
            Some("success") => Some(true),
            Some("failure") => Some(false),
            Some(value) => return Err(anyhow!("Invalid history outcome filter: {value}")),
        };
        let cleanup_filter = match cleanup_filter {
            None => None,
            Some("requested") => Some(true),
            Some("not_requested") => Some(false),
            Some(value) => return Err(anyhow!("Invalid history cleanup filter: {value}")),
        };

        let mut stmt = conn.prepare(
            "SELECT id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_requested, model_id
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
                    OR (?6 = 1 AND transcription_text != '')
                    OR (?6 = 0 AND transcription_text = '')
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
                post_process_requested,
                model_id
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
                post_process_requested,
                model_id
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

    pub fn get_audio_file_path(&self, file_name: &str) -> PathBuf {
        self.recordings_dir.join(file_name)
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
                post_process_requested,
                model_id
             FROM transcription_history
             WHERE id = ?1",
        )?;

        let entry = stmt.query_row([id], Self::map_history_entry).optional()?;

        Ok(entry)
    }

    pub async fn delete_entry(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;

        // Get the entry to find the file name
        if let Some(entry) = self.get_entry_by_id(id).await? {
            // Delete the audio file first
            let file_path = self.get_audio_file_path(&entry.file_name);
            if file_path.exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to delete audio file {}: {}", entry.file_name, e);
                    // Continue with database deletion even if file deletion fails
                }
            }
        }

        // Delete from database
        conn.execute(
            "DELETE FROM transcription_history WHERE id = ?1",
            params![id],
        )?;

        debug!("Deleted history entry with id: {}", id);

        // Emit history updated event
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
                post_process_requested BOOLEAN NOT NULL DEFAULT 0,
                model_id TEXT
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
                post_process_requested,
                model_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                format!("handy-{}.wav", timestamp),
                timestamp,
                false,
                format!("Recording {}", timestamp),
                text,
                post_processed,
                Option::<String>::None,
                false,
                model_id,
            ],
        )
        .expect("insert history entry");
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
                "legacy text",
                Option::<String>::None,
                Option::<String>::None,
                false,
            ],
        )
        .expect("insert pre-model history row");

        Migrations::new(MIGRATIONS.to_vec())
            .to_latest(&mut conn)
            .expect("apply model history migration");

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("load migrated history")
            .expect("migrated entry exists");
        assert_eq!(entry.transcription_text, "legacy text");
        assert!(entry.model_id.is_none());
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
        )
        .expect("update cleanup without retranscribing");

        assert_eq!(entry.transcription_text, "raw transcript");
        assert_eq!(entry.post_processed_text.as_deref(), Some("new cleanup"));
        assert_eq!(entry.post_process_prompt.as_deref(), Some("new prompt"));
        assert!(entry.post_process_requested);
        assert_eq!(entry.model_id.as_deref(), Some("whisper-large-v3-turbo"));
    }

    #[test]
    fn retry_cleanup_rejects_missing_history_entry() {
        let conn = setup_conn();
        let error =
            HistoryManager::update_cleanup_with_conn(&conn, 99, "new cleanup".to_string(), None)
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
}
