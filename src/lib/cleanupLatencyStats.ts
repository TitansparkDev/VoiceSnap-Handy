export const CLEANUP_START_MARKER = "Starting LLM post-processing";

const CLEANUP_COMPLETION_MARKERS = [
  "LLM post-processing succeeded",
  "Structured output post-processing succeeded",
  "Apple Intelligence post-processing succeeded",
  "LLM post-processing failed",
  "Apple Intelligence post-processing failed",
] as const;

export const isCleanupStartLog = (message: string): boolean =>
  message.includes(CLEANUP_START_MARKER);

export const isCleanupCompletionLog = (message: string): boolean =>
  CLEANUP_COMPLETION_MARKERS.some((marker) => message.includes(marker));

const percentileNearestRank = (
  sorted: number[],
  percentile: number,
): number => {
  if (sorted.length === 0) return 0;
  const rank = Math.max(1, Math.ceil(percentile * sorted.length));
  return sorted[Math.min(sorted.length - 1, rank - 1)];
};

export interface CleanupLatencySummary {
  count: number;
  latestMs: number;
  p50Ms: number;
  p95Ms: number;
}

export const summarizeCleanupLatencies = (
  samplesMs: readonly number[],
): CleanupLatencySummary | null => {
  const valid = samplesMs.filter(
    (sample) => Number.isFinite(sample) && sample >= 0,
  );
  if (valid.length === 0) return null;

  const sorted = [...valid].sort((a, b) => a - b);
  return {
    count: sorted.length,
    latestMs: valid[valid.length - 1],
    p50Ms: percentileNearestRank(sorted, 0.5),
    p95Ms: percentileNearestRank(sorted, 0.95),
  };
};
