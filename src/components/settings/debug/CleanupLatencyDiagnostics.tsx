import React, { useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { SettingContainer } from "../../ui/SettingContainer";
import {
  isCleanupCompletionLog,
  isCleanupStartLog,
  summarizeCleanupLatencies,
} from "../../../lib/cleanupLatencyStats";

interface LogEventPayload {
  message: string;
}

const MAX_SAMPLES = 100;

export const CleanupLatencyDiagnostics: React.FC = () => {
  const { t } = useTranslation();
  const [samplesMs, setSamplesMs] = useState<number[]>([]);
  const startedAtRef = useRef<number | null>(null);

  useEffect(() => {
    const unlisten = listen<LogEventPayload>("log://log", (event) => {
      const message = event.payload.message;

      if (isCleanupStartLog(message)) {
        startedAtRef.current = performance.now();
        return;
      }

      if (!isCleanupCompletionLog(message) || startedAtRef.current === null) {
        return;
      }

      const elapsedMs = Math.max(0, performance.now() - startedAtRef.current);
      startedAtRef.current = null;
      setSamplesMs((current) => {
        const next = current.concat(elapsedMs);
        return next.length > MAX_SAMPLES
          ? next.slice(next.length - MAX_SAMPLES)
          : next;
      });
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const summary = useMemo(
    () => summarizeCleanupLatencies(samplesMs),
    [samplesMs],
  );

  const formatMs = (value: number) => `${Math.round(value)} ms`;

  return (
    <SettingContainer
      title={t("settings.debug.cleanupLatency.title")}
      description={t("settings.debug.cleanupLatency.description")}
      descriptionMode="tooltip"
      grouped={true}
      layout="stacked"
    >
      {summary ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 w-full">
          <Metric
            label={t("settings.debug.cleanupLatency.latest")}
            value={formatMs(summary.latestMs)}
          />
          <Metric
            label={t("settings.debug.cleanupLatency.p50")}
            value={formatMs(summary.p50Ms)}
          />
          <Metric
            label={t("settings.debug.cleanupLatency.p95")}
            value={formatMs(summary.p95Ms)}
          />
          <Metric
            label={t("settings.debug.cleanupLatency.samples")}
            value={String(summary.count)}
          />
        </div>
      ) : (
        <p className="text-sm text-mid-gray">
          {t("settings.debug.cleanupLatency.empty")}
        </p>
      )}
    </SettingContainer>
  );
};

const Metric: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div className="rounded-lg border border-mid-gray/20 bg-mid-gray/5 p-3">
    <div className="text-xs text-mid-gray">{label}</div>
    <div className="mt-1 font-mono text-sm font-semibold tabular-nums">
      {value}
    </div>
  </div>
);
