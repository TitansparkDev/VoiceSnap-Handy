export type RuntimeRecoveryReason = "gpu_cpu_fallback" | "auto_cpu_fallback";

export interface RuntimeDiagnosticFields {
  backend: string | null;
  device: string | null;
  saved_accelerator: string | null;
  recommended_backend: string | null;
  recommended_device: string | null;
}

export interface RuntimeRecoveryDiagnostic {
  reason: RuntimeRecoveryReason;
  recommendedLabel: string | null;
}

export const formatRecommendedRuntime = (
  entry: RuntimeDiagnosticFields,
): string | null => {
  return (
    [entry.recommended_backend, entry.recommended_device]
      .filter((value): value is string => Boolean(value))
      .join(" · ") || null
  );
};

/**
 * Classify the user-visible recovery case for a completed transcribe.cpp run.
 * History persists the saved preference separately from the recommended and
 * actual runtime values, so a CPU result can explain fallback without implying
 * that Handy rewrote the user's setting.
 */
export const getRuntimeRecoveryDiagnostic = (
  entry: RuntimeDiagnosticFields,
): RuntimeRecoveryDiagnostic | null => {
  if (entry.backend?.trim().toLowerCase() !== "cpu") return null;

  const savedAccelerator = entry.saved_accelerator?.trim().toLowerCase();
  if (savedAccelerator !== "gpu" && savedAccelerator !== "auto") return null;

  return {
    reason:
      savedAccelerator === "gpu" ? "gpu_cpu_fallback" : "auto_cpu_fallback",
    recommendedLabel: formatRecommendedRuntime(entry),
  };
};
