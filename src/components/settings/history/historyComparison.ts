export interface HistoryTextPair {
  transcription_text: string;
  post_processed_text?: string | null;
}

export interface HistoryComparison {
  raw: string;
  final: string;
}

/**
 * Return a compare view only when post-processing produced meaningful text that
 * differs from the raw transcription. Whitespace-only or unchanged provider
 * output stays on the normal single-text path.
 */
export const getHistoryComparison = (
  entry: HistoryTextPair,
): HistoryComparison | null => {
  const raw = entry.transcription_text.trim();
  const final = entry.post_processed_text?.trim() ?? "";

  if (!raw || !final || raw === final) return null;

  return {
    raw: entry.transcription_text,
    final: entry.post_processed_text!,
  };
};
