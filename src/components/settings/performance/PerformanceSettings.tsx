import React, { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

interface PerformanceStage {
  name: string;
  duration_ms: number;
}

interface PerformanceSample {
  session_id: number;
  outcome: string;
  cold_start: boolean;
  model_id: string | null;
  engine_type: string | null;
  language: string | null;
  backend: string | null;
  device: string | null;
  cleanup_mode: string;
  insertion_mode: string;
  recording_ms: number | null;
  first_partial_ms: number | null;
  stages: PerformanceStage[];
}

interface StagePercentiles {
  stage: string;
  sample_count: number;
  p50_ms: number;
  p95_ms: number;
}

interface PerformanceWindowSummary {
  window: number;
  sample_count: number;
  stages: StagePercentiles[];
}

interface PerformanceSnapshot {
  sample_count: number;
  latest: PerformanceSample | null;
  windows: PerformanceWindowSummary[];
}

const PHASE_STAGE_NAMES = new Set([
  "capture_duration",
  "transcription_total",
  "cleanup_total",
  "paste_total",
]);

const formatStageName = (stage: string) =>
  stage
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const formatMs = (value: number | null) =>
  value == null ? "—" : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value} ms`;

export const PerformanceSettings: React.FC = () => {
  const { t } = useTranslation();
  const [snapshot, setSnapshot] = useState<PerformanceSnapshot | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const next = await invoke<PerformanceSnapshot>("get_performance_diagnostics");
      setSnapshot(next);
    } catch (error) {
      console.error("Failed to load performance diagnostics:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const slowestStage = useMemo(() => {
    const latest = snapshot?.latest;
    if (!latest || latest.stages.length === 0) return null;
    const phaseStages = latest.stages.filter((stage) => PHASE_STAGE_NAMES.has(stage.name));
    const candidates = phaseStages.length > 0 ? phaseStages : latest.stages;
    return candidates.reduce((slowest, stage) =>
      stage.duration_ms > slowest.duration_ms ? stage : slowest,
    );
  }, [snapshot?.latest]);

  const maxStageMs = useMemo(
    () => Math.max(1, ...(snapshot?.latest?.stages.map((stage) => stage.duration_ms) ?? [1])),
    [snapshot?.latest],
  );

  const copyDiagnostics = async () => {
    try {
      const exportText = await invoke<string>("export_performance_diagnostics");
      await navigator.clipboard.writeText(exportText);
      toast.success(
        t("settings.performance.copied", { defaultValue: "Safe diagnostics copied" }),
      );
    } catch (error) {
      toast.error(
        t("settings.performance.copyFailed", { defaultValue: "Could not copy diagnostics" }),
      );
      console.error("Failed to export performance diagnostics:", error);
    }
  };

  const clearDiagnostics = async () => {
    try {
      await invoke("clear_performance_diagnostics");
      await refresh();
      toast.success(
        t("settings.performance.cleared", { defaultValue: "Performance history cleared" }),
      );
    } catch (error) {
      console.error("Failed to clear performance diagnostics:", error);
    }
  };

  const latest = snapshot?.latest;

  return (
    <div className="max-w-3xl w-full mx-auto space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">
            {t("settings.performance.title", { defaultValue: "Diagnostics / Performance" })}
          </h2>
          <p className="text-sm text-mid-gray mt-1">
            {t("settings.performance.description", {
              defaultValue:
                "Timing-only session diagnostics. Transcript, audio, clipboard, process-path and window-title content are not stored here.",
            })}
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            className="px-3 py-2 rounded-lg border border-mid-gray/20 text-sm hover:bg-mid-gray/10 disabled:opacity-50"
            disabled={!snapshot?.sample_count}
            onClick={copyDiagnostics}
          >
            {t("settings.performance.copy", { defaultValue: "Copy diagnostics" })}
          </button>
          <button
            type="button"
            className="px-3 py-2 rounded-lg border border-mid-gray/20 text-sm hover:bg-mid-gray/10 disabled:opacity-50"
            disabled={!snapshot?.sample_count}
            onClick={clearDiagnostics}
          >
            {t("settings.performance.clear", { defaultValue: "Clear" })}
          </button>
        </div>
      </div>

      {loading && !snapshot ? (
        <div className="rounded-lg border border-mid-gray/20 p-4 text-sm text-mid-gray">
          {t("settings.performance.loading", { defaultValue: "Loading diagnostics…" })}
        </div>
      ) : !latest ? (
        <div className="rounded-lg border border-mid-gray/20 p-4 text-sm text-mid-gray">
          {t("settings.performance.empty", {
            defaultValue: "No performance samples yet. Complete or cancel a dictation to create one.",
          })}
        </div>
      ) : (
        <>
          <section className="rounded-lg border border-mid-gray/20 p-4 space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="font-medium">
                  {t("settings.performance.latest", { defaultValue: "Latest session" })}
                </h3>
                <p className="text-xs text-mid-gray mt-1">
                  #{latest.session_id} · {latest.outcome} · {latest.cold_start ? "cold start" : "warm"}
                </p>
              </div>
              {slowestStage && (
                <div className="text-xs text-end">
                  <div className="text-mid-gray">
                    {t("settings.performance.slowest", { defaultValue: "Slowest stage" })}
                  </div>
                  <div className="font-medium">
                    {formatStageName(slowestStage.name)} · {formatMs(slowestStage.duration_ms)}
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">Model</div>
                <div className="truncate" title={latest.model_id ?? undefined}>{latest.model_id ?? "—"}</div>
              </div>
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">Engine</div>
                <div>{latest.engine_type ?? "—"}</div>
              </div>
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">Backend / device</div>
                <div className="truncate" title={[latest.backend, latest.device].filter(Boolean).join(" / ") || undefined}>
                  {[latest.backend, latest.device].filter(Boolean).join(" / ") || "—"}
                </div>
              </div>
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">Recording</div>
                <div>{formatMs(latest.recording_ms)}</div>
              </div>
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">First partial</div>
                <div>{formatMs(latest.first_partial_ms)}</div>
              </div>
              <div className="rounded-md bg-mid-gray/10 p-2">
                <div className="text-xs text-mid-gray">Mode</div>
                <div className="truncate" title={`${latest.insertion_mode} / ${latest.cleanup_mode}`}>
                  {latest.insertion_mode} / {latest.cleanup_mode}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {latest.stages.map((stage) => {
                const isSlowest = slowestStage?.name === stage.name;
                const width = Math.max(3, Math.round((stage.duration_ms / maxStageMs) * 100));
                return (
                  <div key={stage.name} className={`rounded-md border p-2 ${isSlowest ? "border-logo-primary" : "border-mid-gray/15"}`}>
                    <div className="flex justify-between gap-2 text-xs mb-1">
                      <span className={isSlowest ? "font-semibold" : ""}>{formatStageName(stage.name)}</span>
                      <span>{formatMs(stage.duration_ms)}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-mid-gray/10 overflow-hidden">
                      <div className="h-full rounded-full bg-logo-primary/70" style={{ width: `${width}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-lg border border-mid-gray/20 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">p50 / p95</h3>
              <span className="text-xs text-mid-gray">{snapshot?.sample_count ?? 0} retained / 200 max</span>
            </div>
            <div className="space-y-3">
              {snapshot?.windows.map((windowSummary) => (
                <div key={windowSummary.window} className="rounded-md bg-mid-gray/5 p-3">
                  <div className="text-sm font-medium mb-2">
                    Last {windowSummary.window} · {windowSummary.sample_count} sample{windowSummary.sample_count === 1 ? "" : "s"}
                  </div>
                  {windowSummary.stages.length === 0 ? (
                    <div className="text-xs text-mid-gray">No stage timings yet.</div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      {windowSummary.stages.map((stage) => (
                        <div key={stage.stage} className="flex justify-between gap-3 border-b border-mid-gray/10 py-1">
                          <span className="truncate" title={formatStageName(stage.stage)}>{formatStageName(stage.stage)}</span>
                          <span className="shrink-0">p50 {formatMs(stage.p50_ms)} · p95 {formatMs(stage.p95_ms)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
};
