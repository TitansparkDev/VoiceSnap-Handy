use crate::settings::{VocabularySettingsV1, VocabularyEntry, VocabularyReplacement};
use natural::phonetics::soundex;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use strsim::levenshtein;

const VOCABULARY_PROMPT_MAX_ENTRIES: usize = 64;
const VOCABULARY_PROMPT_MAX_CHARS: usize = 2048;
const VOCABULARY_TERM_MAX_CHARS: usize = 80;
const VOCABULARY_MAX_RULES: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VocabularyCorrectionMetadata {
    pub alias_replacements: usize,
    pub scoped_replacements: usize,
    pub fuzzy_applied: bool,
}

impl VocabularyCorrectionMetadata {
    pub fn applied(&self) -> bool {
        self.alias_replacements > 0 || self.scoped_replacements > 0 || self.fuzzy_applied
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyCorrectionResult {
    pub text: String,
    pub metadata: VocabularyCorrectionMetadata,
}

/// Builds an n-gram string by cleaning and concatenating words
///
/// Strips punctuation from each word, lowercases, and joins without spaces.
/// This allows matching "Charge B" against "ChargeBee".
fn build_ngram(words: &[&str]) -> String {
    words
        .iter()
        .map(|w| build_match_key(w))
        .collect::<Vec<_>>()
        .concat()
}

fn build_match_key(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

struct CustomWordMatchKey {
    word_index: usize,
    key: String,
}

fn build_custom_word_match_keys(word: &str, word_index: usize) -> Vec<CustomWordMatchKey> {
    let primary_key = build_match_key(word);
    let mut keys = Vec::with_capacity(2);

    // The fallback matcher is intentionally limited to ASCII terms. Its
    // whitespace tokenization and Soundex scoring are not suitable for CJK
    // scripts. Unicode custom words remain available to models that accept
    // them as native decode prompts; they are simply skipped by this fallback.
    if is_supported_fuzzy_key(&primary_key) {
        keys.push(CustomWordMatchKey {
            word_index,
            key: primary_key.clone(),
        });
    }

    if word.contains('&') {
        let expanded_key = build_match_key(&word.replace('&', " and "));
        if is_supported_fuzzy_key(&expanded_key) && expanded_key != primary_key {
            keys.push(CustomWordMatchKey {
                word_index,
                key: expanded_key,
            });
        }
    }

    keys
}

fn is_supported_fuzzy_key(key: &str) -> bool {
    !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric())
}

fn supports_soundex(key: &str) -> bool {
    !key.is_empty() && key.chars().all(|c| c.is_ascii_alphabetic())
}

/// Finds the best matching custom word for a candidate string
///
/// Uses Levenshtein distance and Soundex phonetic matching to find
/// the best match above the given threshold.
///
/// # Arguments
/// * `candidate` - The cleaned/lowercased candidate string to match
/// * `custom_words` - Original custom words (for returning the replacement)
/// * `custom_word_match_keys` - Normalized custom-word keys for comparison
/// * `threshold` - Maximum similarity score to accept
///
/// # Returns
/// The best matching custom word and its score, if any match was found
fn find_best_match<'a>(
    candidate: &str,
    custom_words: &'a [String],
    custom_word_match_keys: &[CustomWordMatchKey],
    threshold: f64,
) -> Option<(&'a String, f64)> {
    let candidate_len = candidate.chars().count();
    if !is_supported_fuzzy_key(candidate) || candidate_len > 50 {
        return None;
    }

    // The user-facing threshold predates the richer vocabulary contract and can
    // be set very high in old stores. Cap fuzzy edits to a conservative bound;
    // exact matches still work regardless of this cap.
    let safe_threshold = threshold.clamp(0.0, 0.34);
    let mut best_match: Option<&String> = None;
    let mut best_index: Option<usize> = None;
    let mut best_score = f64::MAX;
    let mut second_best_score = f64::MAX;

    for custom_word_key in custom_word_match_keys {
        let custom_word_len = custom_word_key.key.chars().count();
        let exact_match = candidate == custom_word_key.key;

        // Short/common tokens are only eligible for exact matching. Fuzzy edits
        // such as "a" -> "AI" or "in" -> "Inn" are too risky to apply.
        if !exact_match && (candidate_len < 4 || custom_word_len < 4) {
            continue;
        }

        let len_diff = candidate_len.abs_diff(custom_word_len) as f64;
        let max_len = candidate_len.max(custom_word_len) as f64;
        let max_allowed_diff = (max_len * 0.25).max(1.0);
        if !exact_match && len_diff > max_allowed_diff {
            continue;
        }

        let levenshtein_score = if exact_match {
            0.0
        } else {
            levenshtein(candidate, &custom_word_key.key) as f64 / max_len
        };
        let phonetic_match = !exact_match
            && supports_soundex(candidate)
            && supports_soundex(&custom_word_key.key)
            && soundex(candidate, &custom_word_key.key);
        let combined_score = if phonetic_match {
            levenshtein_score * 0.3
        } else {
            levenshtein_score
        };

        if !exact_match && combined_score > safe_threshold {
            continue;
        }

        if combined_score < best_score {
            if best_index != Some(custom_word_key.word_index) {
                second_best_score = best_score;
            }
            best_match = Some(&custom_words[custom_word_key.word_index]);
            best_index = Some(custom_word_key.word_index);
            best_score = combined_score;
        } else if best_index != Some(custom_word_key.word_index)
            && combined_score < second_best_score
        {
            second_best_score = combined_score;
        }
    }

    // Ambiguous non-exact fuzzy candidates fail closed instead of depending on
    // list order. A clear margin is required between the best two terms.
    if best_score > 0.0 && second_best_score.is_finite() && second_best_score - best_score < 0.08 {
        return None;
    }

    best_match.map(|m| (m, best_score))
}

/// Applies custom word corrections to transcribed text using fuzzy matching
///
/// This function corrects words in the input text by finding the best matches
/// from a list of custom words using a combination of:
/// - Levenshtein distance for string similarity
/// - Soundex phonetic matching for pronunciation similarity
/// - N-gram matching for multi-word speech artifacts (e.g., "Charge B" -> "ChargeBee")
///
/// # Arguments
/// * `text` - The input text to correct
/// * `custom_words` - List of custom words to match against
/// * `threshold` - Maximum similarity score to accept (0.0 = exact match, 1.0 = any match)
///
/// # Returns
/// The corrected text with custom words applied
pub fn apply_custom_words(text: &str, custom_words: &[String], threshold: f64) -> String {
    if custom_words.is_empty() {
        return text.to_string();
    }

    // Pre-compute normalized comparison keys to avoid repeated allocations.
    let mut seen_words = HashSet::new();
    let custom_word_match_keys: Vec<CustomWordMatchKey> = custom_words
        .iter()
        .enumerate()
        .filter(|(_, word)| seen_words.insert(build_match_key(word)))
        .flat_map(|(index, word)| build_custom_word_match_keys(word, index))
        .collect();

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let mut best_match: Option<(usize, &String, f64)> = None;

        // Consider n-grams up to three words and choose the closest match. A
        // longest-first match can consume a following ordinary word when both
        // candidates happen to share a Soundex code (for example,
        // "Charge B, che" matching "ChargeBee").
        for n in (1..=3).rev() {
            if i + n > words.len() {
                continue;
            }

            let ngram_words = &words[i..i + n];
            // Do not consume across a punctuation boundary. In
            // "Charge B, che", the comma closes the candidate at "B,".
            if ngram_words[..n.saturating_sub(1)]
                .iter()
                .any(|word| !extract_punctuation(word).1.is_empty())
            {
                continue;
            }
            let ngram = build_ngram(ngram_words);

            if let Some((replacement, score)) =
                find_best_match(&ngram, custom_words, &custom_word_match_keys, threshold)
            {
                let is_better = best_match
                    .as_ref()
                    .is_none_or(|(_, _, best_score)| score < *best_score);
                if is_better {
                    best_match = Some((n, replacement, score));
                }
            }
        }

        if let Some((n, replacement, _)) = best_match {
            let ngram_words = &words[i..i + n];
            // Extract punctuation from first and last words of the n-gram.
            let (prefix, _) = extract_punctuation(ngram_words[0]);
            let (_, suffix) = extract_punctuation(ngram_words[n - 1]);

            // Preserve case from first word.
            let corrected = preserve_case_pattern(ngram_words[0], replacement);

            result.push(format!("{}{}{}", prefix, corrected, suffix));
            i += n;
        } else {
            result.push(words[i].to_string());
            i += 1;
        }
    }

    result.join(" ")
}

fn normalized_language(language: &str) -> Option<String> {
    let normalized = language.trim().to_lowercase();
    if normalized.is_empty() || normalized == "auto" {
        None
    } else {
        Some(normalized.replace('_', "-"))
    }
}

fn language_scope_matches(scope: Option<&str>, output_language: Option<&str>) -> bool {
    let Some(scope) = scope.and_then(normalized_language) else {
        return true;
    };
    let Some(output) = output_language.and_then(normalized_language) else {
        return false;
    };

    scope == output
        || scope.split('-').next() == output.split('-').next()
}

fn sanitized_prompt_term(value: &str) -> Option<String> {
    let without_controls: String = value.chars().filter(|c| !c.is_control()).collect();
    let compact = without_controls.split_whitespace().collect::<Vec<_>>().join(" ");
    let bounded: String = compact.chars().take(VOCABULARY_TERM_MAX_CHARS).collect();
    (!bounded.is_empty()).then_some(bounded)
}

fn safe_rule_term(value: &str) -> Option<String> {
    if value.chars().any(|c| c.is_control()) {
        return None;
    }
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.chars().count() > VOCABULARY_TERM_MAX_CHARS {
        return None;
    }
    Some(trimmed.to_string())
}

/// Build bounded, control-character-free model context. Only canonical written
/// forms are sent; spoken aliases and replacement sources remain local so prompt
/// injection through those fields is impossible.
pub fn build_vocabulary_prompt(
    vocabulary: &VocabularySettingsV1,
    legacy_custom_words: &[String],
    language: Option<&str>,
) -> Option<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();
    let mut used_chars = 0usize;

    let mut push_term = |term: &str| {
        if terms.len() >= VOCABULARY_PROMPT_MAX_ENTRIES {
            return;
        }
        let Some(term) = sanitized_prompt_term(term) else {
            return;
        };
        let dedupe_key = term.to_lowercase();
        if !seen.insert(dedupe_key) {
            return;
        }
        let separator_chars = usize::from(!terms.is_empty()) * 2;
        let term_chars = term.chars().count();
        if used_chars + separator_chars + term_chars > VOCABULARY_PROMPT_MAX_CHARS {
            return;
        }
        used_chars += separator_chars + term_chars;
        terms.push(term);
    };

    for entry in vocabulary.entries.iter().take(VOCABULARY_MAX_RULES) {
        if entry.enabled && language_scope_matches(entry.language.as_deref(), language) {
            push_term(&entry.written);
        }
    }
    // Once rich entries exist they are authoritative. Migration copies every
    // legacy word into this list, so continuing to apply `custom_words` as a
    // second source would make a migrated entry impossible to disable/remove.
    if vocabulary.entries.is_empty() {
        for word in legacy_custom_words {
            push_term(word);
        }
    }

    (!terms.is_empty()).then(|| terms.join(", "))
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn exact_rule_replace(
    text: &str,
    from: &str,
    to: &str,
    case_sensitive: bool,
    preserve_punctuation: bool,
) -> (String, usize) {
    let pattern = if case_sensitive {
        regex::escape(from)
    } else {
        format!("(?i:{})", regex::escape(from))
    };
    let Ok(regex) = Regex::new(&pattern) else {
        return (text.to_string(), 0);
    };

    let mut result = String::with_capacity(text.len());
    let mut cursor = 0usize;
    let mut replacements = 0usize;
    for matched in regex.find_iter(text) {
        let left = text[..matched.start()].chars().next_back();
        let right = text[matched.end()..].chars().next();
        if left.is_some_and(is_word_char) || right.is_some_and(is_word_char) {
            continue;
        }
        if !preserve_punctuation
            && (left.is_some_and(|c| !c.is_whitespace() && !is_word_char(c))
                || right.is_some_and(|c| !c.is_whitespace() && !is_word_char(c)))
        {
            continue;
        }

        result.push_str(&text[cursor..matched.start()]);
        result.push_str(to);
        cursor = matched.end();
        replacements += 1;
    }

    if replacements == 0 {
        return (text.to_string(), 0);
    }
    result.push_str(&text[cursor..]);
    (result, replacements)
}

fn apply_replacement_rule(
    text: &str,
    rule: &VocabularyReplacement,
) -> (String, usize) {
    let (Some(from), Some(to)) = (safe_rule_term(&rule.from), safe_rule_term(&rule.to)) else {
        return (text.to_string(), 0);
    };
    exact_rule_replace(
        text,
        &from,
        &to,
        rule.case_sensitive.unwrap_or(false),
        rule.preserve_punctuation.unwrap_or(true),
    )
}

fn apply_alias_rule(text: &str, entry: &VocabularyEntry) -> (String, usize) {
    let Some(alias) = entry.spoken_alias.as_deref().and_then(safe_rule_term) else {
        return (text.to_string(), 0);
    };
    let Some(written) = safe_rule_term(&entry.written) else {
        return (text.to_string(), 0);
    };
    if alias.eq_ignore_ascii_case(&written) {
        return (text.to_string(), 0);
    }
    exact_rule_replace(
        text,
        &alias,
        &written,
        entry.case_sensitive.unwrap_or(false),
        entry.preserve_punctuation.unwrap_or(true),
    )
}

/// Apply deterministic scoped replacements and aliases first, then the bounded
/// fuzzy fallback when the model did not already receive vocabulary context.
/// The result carries counts only; no transcript or vocabulary contents are
/// needed to attribute the operation in logs/history metadata.
pub fn apply_vocabulary_corrections(
    text: &str,
    vocabulary: &VocabularySettingsV1,
    legacy_custom_words: &[String],
    threshold: f64,
    output_language: &OutputLanguageEvidence,
    allow_fuzzy: bool,
) -> VocabularyCorrectionResult {
    let output_language = output_language.language();
    let mut current = text.to_string();
    let mut metadata = VocabularyCorrectionMetadata::default();
    let mut seen_rules = HashSet::new();

    for rule in vocabulary.replacements.iter().take(VOCABULARY_MAX_RULES) {
        if !rule.enabled || !language_scope_matches(rule.language.as_deref(), output_language) {
            continue;
        }
        let dedupe_key = format!(
            "replacement:{}:{}:{}",
            rule.from.to_lowercase(),
            rule.to.to_lowercase(),
            rule.language.as_deref().unwrap_or("").to_lowercase()
        );
        if !seen_rules.insert(dedupe_key) {
            continue;
        }
        let (updated, count) = apply_replacement_rule(&current, rule);
        current = updated;
        metadata.scoped_replacements += count;
    }

    for entry in vocabulary.entries.iter().take(VOCABULARY_MAX_RULES) {
        if !entry.enabled || !language_scope_matches(entry.language.as_deref(), output_language) {
            continue;
        }
        let Some(alias) = entry.spoken_alias.as_deref() else {
            continue;
        };
        let dedupe_key = format!(
            "alias:{}:{}:{}",
            alias.to_lowercase(),
            entry.written.to_lowercase(),
            entry.language.as_deref().unwrap_or("").to_lowercase()
        );
        if !seen_rules.insert(dedupe_key) {
            continue;
        }
        let (updated, count) = apply_alias_rule(&current, entry);
        current = updated;
        metadata.alias_replacements += count;
    }

    if allow_fuzzy {
        let mut fuzzy_words = Vec::new();
        let mut seen_words = HashSet::new();
        for entry in vocabulary.entries.iter().take(VOCABULARY_MAX_RULES) {
            if !entry.enabled || !language_scope_matches(entry.language.as_deref(), output_language) {
                continue;
            }
            let Some(written) = safe_rule_term(&entry.written) else {
                continue;
            };
            if seen_words.insert(build_match_key(&written)) {
                fuzzy_words.push(written);
            }
        }
        if vocabulary.entries.is_empty() {
            for word in legacy_custom_words {
                let Some(word) = safe_rule_term(word) else {
                    continue;
                };
                if seen_words.insert(build_match_key(&word)) {
                    fuzzy_words.push(word);
                }
            }
        }

        let corrected = apply_custom_words(&current, &fuzzy_words, threshold);
        if corrected != current {
            metadata.fuzzy_applied = true;
            current = corrected;
        }
    }

    VocabularyCorrectionResult {
        text: current,
        metadata,
    }
}

/// Preserves the case pattern of the original word when applying a replacement
fn preserve_case_pattern(original: &str, replacement: &str) -> String {
    if original.chars().all(|c| c.is_uppercase()) {
        replacement.to_uppercase()
    } else if original.chars().next().is_some_and(|c| c.is_uppercase()) {
        let mut chars: Vec<char> = replacement.chars().collect();
        if let Some(first_char) = chars.get_mut(0) {
            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
        }
        chars.into_iter().collect()
    } else {
        replacement.to_string()
    }
}

/// Extracts punctuation prefix and suffix from a word
fn extract_punctuation(word: &str) -> (&str, &str) {
    // String slices use byte offsets. Derive both boundaries from char_indices
    // so multibyte punctuation such as `。` and `「」` can never be split.
    let prefix_end = word
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(index, _)| index)
        .unwrap_or(word.len());
    let suffix_start = word
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(index, c)| index + c.len_utf8())
        .unwrap_or(0);

    let prefix = if prefix_end > 0 {
        &word[..prefix_end]
    } else {
        ""
    };

    let suffix = if suffix_start < word.len() {
        &word[suffix_start..]
    } else {
        ""
    };

    (prefix, suffix)
}

/// Evidence for the language of the text being cleaned.
///
/// This intentionally describes the transcription output, not Handy's UI
/// language. Unknown output languages fail closed: built-in filler removal is
/// skipped rather than applying a language profile speculatively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputLanguageEvidence {
    UserSelected(String),
    ModelConstrained(String),
    /// The transcription model itself identified the language (audio-based
    /// LID, e.g. Whisper in auto mode).
    ModelDetected(String),
    /// Detected from the transcribed text with high confidence, constrained to
    /// the model's supported languages. Weakest accepted evidence.
    TextDetected(String),
    TranslatedToEnglish,
    Unknown,
}

impl OutputLanguageEvidence {
    fn language(&self) -> Option<&str> {
        match self {
            Self::UserSelected(language)
            | Self::ModelConstrained(language)
            | Self::ModelDetected(language)
            | Self::TextDetected(language) => Some(language),
            Self::TranslatedToEnglish => Some("en"),
            Self::Unknown => None,
        }
    }
}

/// Filler tokens that are not lexical words in any language Handy's models can
/// output, so removing them cannot corrupt text regardless of the (possibly
/// unknown) output language. Kept deliberately conservative: anything that is a
/// real word somewhere ("um" pt/de, "ha" es, "ah"/"eh" interjections, "mm"
/// millimetres) belongs in the language-gated lists instead.
const UNIVERSAL_FILLER_WORDS: &[&str] = &[
    "uh", "uhm", "umm", "uhh", "uhhh", "ehh", "ehm", "ahm", "hmm", "hm", "mmm", "хм", "ммм",
];

/// Filler words that are only safe to remove with evidence for the output
/// language, because the same token is a real word elsewhere (e.g. Portuguese
/// "um" = "a/an", German "um" = "at/around", Spanish "ha" = "has").
fn gated_filler_words_for_language(lang: &str) -> &'static [&'static str] {
    let base_lang = lang.split(&['-', '_'][..]).next().unwrap_or(lang);

    match base_lang {
        "en" => &["um", "ah", "eh", "ha"],
        "de" => &["äh", "ähm"],
        "fr" => &["euh"],
        _ => &[],
    }
}

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Collapses repeated words (3+ repetitions) to a single instance.
/// E.g., "wh wh wh wh" -> "wh", "I I I I" -> "I"
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        if word_lower.chars().all(|c| c.is_alphabetic()) {
            // Count consecutive repetitions (case-insensitive)
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }

            // If 3+ repetitions, collapse to single instance
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Removes filler words from transcription output when enabled.
///
/// Built-in removal is two-tiered: [`UNIVERSAL_FILLER_WORDS`] apply regardless
/// of language evidence, while [`gated_filler_words_for_language`] tokens are
/// only removed when the output language is known. A custom list is an
/// explicit user override and replaces both tiers without requiring language
/// evidence. `Some(empty vec)` disables removal, preserving the legacy
/// power-user setting. The master toggle takes precedence over both built-in
/// and custom lists.
///
/// # Arguments
/// * `text` - The raw transcription text to filter
/// * `language` - Evidence for the language of the transcription output
/// * `custom_filler_words` - Optional user-provided filler word list. `Some(vec)` overrides
///   language defaults; `Some(empty vec)` disables filtering; `None` uses language defaults.
/// * `enabled` - Whether filler-word removal is enabled
///
/// # Returns
/// The text with configured filler words removed
pub fn remove_filler_words(
    text: &str,
    language: &OutputLanguageEvidence,
    custom_filler_words: &Option<Vec<String>>,
    enabled: bool,
) -> String {
    if !enabled {
        return text.to_string();
    }

    // Build filler patterns from custom list or the built-in tiers
    let patterns: Vec<Regex> = match custom_filler_words {
        Some(words) => words
            .iter()
            .filter_map(|word| Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).ok())
            .collect(),
        None => UNIVERSAL_FILLER_WORDS
            .iter()
            .chain(
                language
                    .language()
                    .map(gated_filler_words_for_language)
                    .unwrap_or_default(),
            )
            .map(|word| Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).unwrap())
            .collect(),
    };

    // Remove filler words
    let mut filtered = text.to_string();
    for pattern in &patterns {
        filtered = pattern.replace_all(&filtered, "").to_string();
    }

    filtered
}

/// Applies non-filler transcription cleanup.
///
/// Kept separate from [`remove_filler_words`] so disabling filler deletion
/// does not also disable the existing repeated-word and whitespace cleanup.
pub fn normalize_transcription_output(text: &str) -> String {
    let mut normalized = collapse_stutters(text);

    // Clean up multiple spaces to single space
    normalized = MULTI_SPACE_PATTERN
        .replace_all(&normalized, " ")
        .to_string();

    // Trim leading/trailing whitespace
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the complete cleanup sequence with an explicitly selected
    /// language. Individual tests below predate the split between filler
    /// removal and non-filler normalization.
    fn filter_transcription_output(
        text: &str,
        language: &str,
        custom_filler_words: &Option<Vec<String>>,
    ) -> String {
        let language = OutputLanguageEvidence::UserSelected(language.to_string());
        let filtered = remove_filler_words(text, &language, custom_filler_words, true);
        normalize_transcription_output(&filtered)
    }

    fn rich_entry(
        written: &str,
        alias: Option<&str>,
        language: Option<&str>,
    ) -> VocabularyEntry {
        VocabularyEntry {
            written: written.to_string(),
            spoken_alias: alias.map(str::to_string),
            language: language.map(str::to_string),
            enabled: true,
            case_sensitive: None,
            preserve_punctuation: None,
        }
    }

    fn vocabulary(
        entries: Vec<VocabularyEntry>,
        replacements: Vec<VocabularyReplacement>,
    ) -> VocabularySettingsV1 {
        VocabularySettingsV1 {
            version: 1,
            entries,
            replacements,
        }
    }

    #[test]
    fn test_apply_custom_words_exact_match() {
        let text = "hello world";
        let custom_words = vec!["Hello".to_string(), "World".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_apply_custom_words_fuzzy_match() {
        let text = "helo wrold";
        let custom_words = vec!["hello".to_string(), "world".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preserve_case_pattern() {
        assert_eq!(preserve_case_pattern("HELLO", "world"), "WORLD");
        assert_eq!(preserve_case_pattern("Hello", "world"), "World");
        assert_eq!(preserve_case_pattern("hello", "WORLD"), "WORLD");
    }

    #[test]
    fn test_extract_punctuation() {
        assert_eq!(extract_punctuation("hello"), ("", ""));
        assert_eq!(extract_punctuation("!hello?"), ("!", "?"));
        assert_eq!(extract_punctuation("...hello..."), ("...", "..."));
    }

    #[test]
    fn test_extract_punctuation_uses_unicode_boundaries() {
        assert_eq!(extract_punctuation("你好。"), ("", "。"));
        assert_eq!(extract_punctuation("「你好」"), ("「", "」"));
        assert_eq!(extract_punctuation("你好！"), ("", "！"));
    }

    #[test]
    fn test_empty_custom_words() {
        let text = "hello world";
        let custom_words = vec![];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_filler_words() {
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "So I was thinking about this");
    }

    #[test]
    fn test_filter_filler_words_case_insensitive() {
        let text = "UHM this is UH a test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "this is a test");
    }

    #[test]
    fn test_filter_filler_words_with_punctuation() {
        let text = "Well, uhm, I think, uh. that's right";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Well, I think, that's right");
    }

    #[test]
    fn test_filter_cleans_whitespace() {
        let text = "Hello    world   test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_filter_trims() {
        let text = "  Hello world  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_combined() {
        let text = "  Uhm, so I was, uh, thinking about this  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "so I was, thinking about this");
    }

    #[test]
    fn test_filter_preserves_valid_text() {
        let text = "This is a completely normal sentence.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "This is a completely normal sentence.");
    }

    #[test]
    fn test_filter_stutter_collapse() {
        let text = "w wh wh wh wh wh wh wh wh wh why";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "w wh why");
    }

    #[test]
    fn test_filter_stutter_short_words() {
        let text = "I I I I think so so so so";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think so");
    }

    #[test]
    fn test_filter_stutter_longer_words() {
        let text = "Check data doc doc doc doc documentation.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Check data doc documentation.");
    }

    #[test]
    fn test_filter_stutter_mixed_case() {
        let text = "No NO no NO no";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "No");
    }

    #[test]
    fn test_filter_stutter_preserves_two_repetitions() {
        let text = "no no is fine";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "no no is fine");
    }

    #[test]
    fn test_filter_english_removes_um() {
        let text = "um I think um this is good";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think this is good");
    }

    #[test]
    fn test_filter_portuguese_preserves_um() {
        // "um" means "a/an" in Portuguese
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_spanish_preserves_ha() {
        // "ha" means "has" in Spanish
        let text = "ha sido un buen día";
        let result = filter_transcription_output(text, "es", &None);
        assert_eq!(result, "ha sido un buen día");
    }

    #[test]
    fn test_filter_language_code_with_region() {
        // "pt-BR" should normalize to "pt"
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt-BR", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_custom_filler_words_override() {
        let custom = Some(vec!["okay".to_string(), "right".to_string()]);
        let text = "okay so I think right this works";
        let result = filter_transcription_output(text, "en", &custom);
        assert_eq!(result, "so I think this works");
    }

    #[test]
    fn test_filter_custom_filler_words_empty_disables() {
        let custom = Some(vec![]);
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &custom);
        // No filler words removed since custom list is empty
        assert_eq!(result, "So uhm I was thinking uh about this");
    }

    #[test]
    fn test_filter_unknown_language_still_removes_universal_fillers() {
        let text = "uh I think uhm this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "I think this works");
    }

    #[test]
    fn test_filter_unknown_language_does_not_remove_um() {
        let text = "um I think this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "um I think this works");
    }

    #[test]
    fn test_filter_unknown_evidence_removes_universal_keeps_gated() {
        let filtered = remove_filler_words(
            "uhh bueno hmm creo que um ha llegado",
            &OutputLanguageEvidence::Unknown,
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&filtered),
            "bueno creo que um ha llegado"
        );

        let cyrillic = remove_filler_words(
            "хм я думаю ммм это работает",
            &OutputLanguageEvidence::Unknown,
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&cyrillic),
            "я думаю это работает"
        );
    }

    #[test]
    fn test_filter_german_gated_fillers_require_evidence() {
        let text = "äh ich glaube ähm das passt";

        let unknown = remove_filler_words(text, &OutputLanguageEvidence::Unknown, &None, true);
        assert_eq!(normalize_transcription_output(&unknown), text);

        let result = filter_transcription_output(text, "de", &None);
        assert_eq!(result, "ich glaube das passt");
    }

    #[test]
    fn test_filter_preserves_millimetre_unit() {
        // "mm" was removed from the filler lists because it eats units.
        let text = "the screw is 5 mm long";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "the screw is 5 mm long");
    }

    #[test]
    fn test_filter_detected_evidence_unlocks_gated_fillers() {
        let model = remove_filler_words(
            "um I think this works",
            &OutputLanguageEvidence::ModelDetected("en".to_string()),
            &None,
            true,
        );
        assert_eq!(normalize_transcription_output(&model), "I think this works");

        let text = remove_filler_words(
            "euh je pense que ça marche",
            &OutputLanguageEvidence::TextDetected("fr".to_string()),
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&text),
            "je pense que ça marche"
        );
    }

    #[test]
    fn test_filter_master_toggle_disables_custom_and_builtin_removal() {
        let text = "um customword I think";
        let language = OutputLanguageEvidence::UserSelected("en".to_string());
        let custom = Some(vec!["customword".to_string()]);

        let result = remove_filler_words(text, &language, &custom, false);

        assert_eq!(result, text);
    }

    #[test]
    fn test_filter_custom_words_apply_without_language_evidence() {
        let custom = Some(vec!["customword".to_string()]);
        let text = "customword should be removed but um should remain";

        let filtered = remove_filler_words(text, &OutputLanguageEvidence::Unknown, &custom, true);
        let result = normalize_transcription_output(&filtered);

        assert_eq!(result, "should be removed but um should remain");
    }

    #[test]
    fn test_apply_custom_words_ngram_two_words() {
        let text = "il cui nome è Charge B, che permette";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChargeBee,"), "unexpected result: {result}");
        assert!(!result.contains("Charge B"));
    }

    #[test]
    fn test_apply_custom_words_ngram_three_words() {
        let text = "use Chat G P T for this";
        let custom_words = vec!["ChatGPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChatGPT"));
    }

    #[test]
    fn test_apply_custom_words_prefers_longer_ngram() {
        let text = "Open AI GPT model";
        let custom_words = vec!["OpenAI".to_string(), "GPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "OpenAI GPT model");
    }

    #[test]
    fn test_apply_custom_words_ngram_preserves_case() {
        let text = "CHARGE B is great";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("CHARGEBEE"));
    }

    #[test]
    fn test_apply_custom_words_ngram_with_spaces_in_custom() {
        // Custom word with space should also match against split words
        let text = "using Mac Book Pro";
        let custom_words = vec!["MacBook Pro".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "using MacBook Pro");
    }

    #[test]
    fn test_apply_custom_words_trailing_number_not_doubled() {
        // Verify that trailing non-alpha chars (like numbers) aren't double-counted
        // between build_ngram stripping them and extract_punctuation capturing them
        let text = "use GPT4 for this";
        let custom_words = vec!["GPT-4".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        // Should NOT produce "GPT-44" (double-counting the trailing 4)
        assert!(
            !result.contains("GPT-44"),
            "got double-counted result: {}",
            result
        );
    }

    #[test]
    fn test_apply_custom_words_matches_ampersand_word() {
        let text = "send it to RD for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_matches_spoken_ampersand_word() {
        let text = "send it to R and D for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_preserves_ampersand_word() {
        let text = "send it to R&D for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_handles_unicode_punctuation() {
        let text = "「Handee。」";
        let custom_words = vec!["Handy".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "「Handy。」");
    }

    #[test]
    fn test_apply_custom_words_skips_cjk_fuzzy_matching() {
        let text = "你好。";
        let custom_words = vec!["你号".to_string()];
        let result = apply_custom_words(text, &custom_words, 1.0);
        assert_eq!(result, text);
    }

    #[test]
    fn vocabulary_short_words_never_fuzzy_overmatch() {
        let custom_words = vec!["AI".to_string(), "Inn".to_string()];
        assert_eq!(apply_custom_words("a is useful", &custom_words, 1.0), "a is useful");
        assert_eq!(apply_custom_words("in here", &custom_words, 1.0), "in here");
        assert_eq!(apply_custom_words("AI is useful", &custom_words, 1.0), "AI is useful");
    }

    #[test]
    fn vocabulary_alias_is_scoped_and_does_not_overmatch() {
        let vocab = vocabulary(vec![rich_entry("OpenAI", Some("open eye"), Some("en"))], vec![]);
        let en = OutputLanguageEvidence::UserSelected("en-US".to_string());
        let fr = OutputLanguageEvidence::UserSelected("fr".to_string());

        let corrected = apply_vocabulary_corrections(
            "use open eye, not open eyesight",
            &vocab,
            &[],
            0.18,
            &en,
            true,
        );
        assert_eq!(corrected.text, "use OpenAI, not open eyesight");
        assert_eq!(corrected.metadata.alias_replacements, 1);

        let wrong_language = apply_vocabulary_corrections(
            "use open eye",
            &vocab,
            &[],
            0.18,
            &fr,
            true,
        );
        assert_eq!(wrong_language.text, "use open eye");
        assert!(!wrong_language.metadata.applied());
    }

    #[test]
    fn vocabulary_cjk_alias_uses_exact_boundaries() {
        let vocab = vocabulary(vec![rich_entry("你好", Some("你号"), Some("zh"))], vec![]);
        let zh = OutputLanguageEvidence::ModelDetected("zh-CN".to_string());

        let corrected = apply_vocabulary_corrections("你号。", &vocab, &[], 1.0, &zh, true);
        assert_eq!(corrected.text, "你好。");

        let embedded = apply_vocabulary_corrections("你号世界", &vocab, &[], 1.0, &zh, true);
        assert_eq!(embedded.text, "你号世界");
    }

    #[test]
    fn vocabulary_respects_enabled_case_and_punctuation_policy() {
        let mut disabled = rich_entry("Handy", Some("hand ee"), None);
        disabled.enabled = false;
        let mut case_sensitive = rich_entry("VoiceSnap", Some("Voice Snap"), None);
        case_sensitive.case_sensitive = Some(true);
        let mut punctuation_sensitive = rich_entry("ChatGPT", Some("chat gpt"), None);
        punctuation_sensitive.preserve_punctuation = Some(false);
        let vocab = vocabulary(vec![disabled, case_sensitive, punctuation_sensitive], vec![]);
        let language = OutputLanguageEvidence::Unknown;

        let result = apply_vocabulary_corrections(
            "hand ee voice snap chat gpt, Voice Snap chat gpt",
            &vocab,
            &[],
            0.18,
            &language,
            false,
        );
        assert_eq!(result.text, "hand ee voice snap chat gpt, VoiceSnap ChatGPT");
        assert_eq!(result.metadata.alias_replacements, 2);
    }

    #[test]
    fn vocabulary_rich_entries_override_legacy_activation_state() {
        let mut entry = rich_entry("Handy", None, None);
        entry.enabled = false;
        let vocab = vocabulary(vec![entry], vec![]);
        let result = apply_vocabulary_corrections(
            "Handee",
            &vocab,
            &["Handy".to_string()],
            1.0,
            &OutputLanguageEvidence::Unknown,
            true,
        );
        assert_eq!(result.text, "Handee");
        assert!(!result.metadata.applied());
        assert!(build_vocabulary_prompt(&vocab, &["Handy".to_string()], None).is_none());
    }

    #[test]
    fn vocabulary_duplicate_rules_are_deterministic() {
        let first = rich_entry("OpenAI", Some("open eye"), None);
        let duplicate = rich_entry("OpenAI", Some("open eye"), None);
        let vocab = vocabulary(vec![first, duplicate], vec![]);
        let result = apply_vocabulary_corrections(
            "open eye",
            &vocab,
            &[],
            0.18,
            &OutputLanguageEvidence::Unknown,
            false,
        );
        assert_eq!(result.text, "OpenAI");
        assert_eq!(result.metadata.alias_replacements, 1);
    }

    #[test]
    fn vocabulary_scoped_replacements_do_not_match_inside_words() {
        let vocab = vocabulary(
            vec![],
            vec![VocabularyReplacement {
                from: "cat".to_string(),
                to: "dog".to_string(),
                language: Some("en".to_string()),
                enabled: true,
                case_sensitive: None,
                preserve_punctuation: None,
            }],
        );
        let en = OutputLanguageEvidence::UserSelected("en".to_string());
        let result = apply_vocabulary_corrections(
            "cat concatenate bobcat cat.",
            &vocab,
            &[],
            0.18,
            &en,
            false,
        );
        assert_eq!(result.text, "dog concatenate bobcat dog.");
        assert_eq!(result.metadata.scoped_replacements, 2);
    }

    #[test]
    fn vocabulary_control_characters_are_sanitized_for_prompts_and_rejected_for_rules() {
        let mut entry = rich_entry("Open\u{0007}AI", Some("open\u{0008}eye"), Some("en"));
        entry.enabled = true;
        let vocab = vocabulary(
            vec![entry],
            vec![VocabularyReplacement {
                from: "bad\u{0001}source".to_string(),
                to: "safe".to_string(),
                language: Some("en".to_string()),
                enabled: true,
                case_sensitive: None,
                preserve_punctuation: None,
            }],
        );

        let prompt = build_vocabulary_prompt(&vocab, &[], Some("en")).unwrap();
        assert_eq!(prompt, "OpenAI");
        assert!(!prompt.chars().any(char::is_control));

        let result = apply_vocabulary_corrections(
            "open eye bad source",
            &vocab,
            &[],
            0.18,
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            false,
        );
        assert_eq!(result.text, "open eye bad source");
        assert!(!result.metadata.applied());
    }

    #[test]
    fn vocabulary_prompt_is_bounded_deduplicated_and_contains_only_written_forms() {
        let mut entries = Vec::new();
        for index in 0..100 {
            entries.push(rich_entry(
                &format!("Term{index}\u{0007}{}", "x".repeat(70)),
                Some("SECRET-ALIAS"),
                Some("en"),
            ));
        }
        entries.push(rich_entry("FrenchOnly", Some("french alias"), Some("fr")));
        entries.push(rich_entry("Duplicate", None, Some("en")));
        entries.push(rich_entry("duplicate", None, Some("en")));
        let vocab = vocabulary(entries, vec![]);
        let prompt = build_vocabulary_prompt(&vocab, &["Legacy\u{0001}Word".to_string()], Some("en-US"))
            .expect("bounded vocabulary prompt");

        assert!(prompt.chars().count() <= VOCABULARY_PROMPT_MAX_CHARS);
        assert!(prompt.split(", ").count() <= VOCABULARY_PROMPT_MAX_ENTRIES);
        assert!(!prompt.chars().any(char::is_control));
        assert!(!prompt.contains("SECRET-ALIAS"));
        assert!(!prompt.contains("FrenchOnly"));
    }
}
