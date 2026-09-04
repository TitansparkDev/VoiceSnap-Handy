use serde_json::Value;

pub use crate::settings::{
    VocabularyEntry as VocabularyEntryV1, VocabularyReplacement as VocabularyReplacementV1,
    VocabularySettingsV1,
};

pub const VOCABULARY_V1_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyDecodeError {
    Malformed,
    UnsupportedVersion,
}

/// Convert Handy's legacy `custom_words` list into the versioned vocabulary
/// representation without normalizing, deduplicating, sanitizing, or otherwise
/// changing any stored word. Safety filtering belongs at the model/correction
/// boundary so migration itself is lossless.
pub fn migrate_legacy_custom_words(legacy_words: &[String]) -> VocabularySettingsV1 {
    VocabularySettingsV1 {
        version: VOCABULARY_V1_VERSION,
        entries: legacy_words
            .iter()
            .cloned()
            .map(|written| VocabularyEntryV1 {
                written,
                spoken_alias: None,
                language: None,
                enabled: true,
                case_sensitive: None,
                preserve_punctuation: None,
            })
            .collect(),
        replacements: Vec::new(),
    }
}

/// Decode a stored versioned vocabulary or migrate the legacy list when the
/// richer key is absent. A present-but-malformed/unsupported richer value is an
/// error rather than silently guessing at user intent; callers that transform a
/// transcript can then fail open to that untouched transcript.
pub fn decode_vocabulary_v1(
    raw_vocabulary: Option<&Value>,
    legacy_words: &[String],
) -> Result<VocabularySettingsV1, VocabularyDecodeError> {
    let Some(raw_vocabulary) = raw_vocabulary else {
        return Ok(migrate_legacy_custom_words(legacy_words));
    };

    let vocabulary = serde_json::from_value::<VocabularySettingsV1>(raw_vocabulary.clone())
        .map_err(|_| VocabularyDecodeError::Malformed)?;
    if vocabulary.version != VOCABULARY_V1_VERSION {
        return Err(VocabularyDecodeError::UnsupportedVersion);
    }

    Ok(vocabulary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_preserves_every_legacy_word_verbatim_and_in_order() {
        let legacy_words = vec![
            "Handy".to_string(),
            "R&D".to_string(),
            "你好".to_string(),
            "punctuation?!".to_string(),
            "duplicate".to_string(),
            "duplicate".to_string(),
            "control\u{0007}kept".to_string(),
            String::new(),
        ];

        let migrated = migrate_legacy_custom_words(&legacy_words);

        assert_eq!(migrated.version, VOCABULARY_V1_VERSION);
        assert!(migrated.replacements.is_empty());
        assert_eq!(
            migrated
                .entries
                .iter()
                .map(|entry| entry.written.clone())
                .collect::<Vec<_>>(),
            legacy_words
        );
        assert!(migrated.entries.iter().all(|entry| {
            entry.enabled
                && entry.spoken_alias.is_none()
                && entry.language.is_none()
                && entry.case_sensitive.is_none()
                && entry.preserve_punctuation.is_none()
        }));
    }

    #[test]
    fn decoder_migrates_only_when_rich_key_is_absent() {
        let legacy_words = vec!["LegacyOne".to_string(), "LegacyTwo".to_string()];
        let decoded = decode_vocabulary_v1(None, &legacy_words).expect("legacy migration");

        assert_eq!(
            decoded
                .entries
                .iter()
                .map(|entry| entry.written.as_str())
                .collect::<Vec<_>>(),
            vec!["LegacyOne", "LegacyTwo"]
        );
    }

    #[test]
    fn malformed_or_future_rich_data_is_rejected_instead_of_partially_applied() {
        let legacy_words = vec!["Handy".to_string()];
        let malformed = serde_json::json!({
            "version": 1,
            "entries": [{ "written": 42 }]
        });
        assert_eq!(
            decode_vocabulary_v1(Some(&malformed), &legacy_words),
            Err(VocabularyDecodeError::Malformed)
        );

        let future = serde_json::json!({
            "version": 99,
            "entries": [{ "written": "Future" }],
            "replacements": []
        });
        assert_eq!(
            decode_vocabulary_v1(Some(&future), &legacy_words),
            Err(VocabularyDecodeError::UnsupportedVersion)
        );
    }
}
