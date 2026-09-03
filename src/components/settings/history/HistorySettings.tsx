import React, { useCallback, useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { readFile } from "@tauri-apps/plugin-fs";
import {
  Check,
  Copy,
  FolderOpen,
  RotateCcw,
  Sparkles,
  Star,
  Trash2,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  commands,
  events,
  type HistoryEntry,
  type HistoryUpdatePayload,
} from "@/bindings";
import { useOsType } from "@/hooks/useOsType";
import { formatDateTime } from "@/utils/dateFormat";
import { AudioPlayer, AudioPlayerGroup } from "../../ui/AudioPlayer";
import { Button } from "../../ui/Button";

const IconButton: React.FC<{
  onClick: () => void;
  title: string;
  disabled?: boolean;
  active?: boolean;
  children: React.ReactNode;
}> = ({ onClick, title, disabled, active, children }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`p-1.5 rounded-md flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed disabled:text-text/20 ${
      active
        ? "text-logo-primary hover:text-logo-primary/80"
        : "text-text/50 hover:text-logo-primary"
    }`}
    title={title}
  >
    {children}
  </button>
);

const PAGE_SIZE = 30;

const localMidnightTimestamp = (value: string, dayOffset = 0): number | null => {
  if (!value) return null;

  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return null;

  const date = new Date(year, month - 1, day + dayOffset);
  return Math.floor(date.getTime() / 1000);
};

interface OpenRecordingsButtonProps {
  onClick: () => void;
  label: string;
}

const OpenRecordingsButton: React.FC<OpenRecordingsButtonProps> = ({
  onClick,
  label,
}) => (
  <Button
    onClick={onClick}
    variant="secondary"
    size="sm"
    className="flex items-center gap-2"
    title={label}
  >
    <FolderOpen className="w-4 h-4" />
    <span>{label}</span>
  </Button>
);

export const HistorySettings: React.FC = () => {
  const { t } = useTranslation();
  const osType = useOsType();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("");
  const [cleanupFilter, setCleanupFilter] = useState("");
  const [modelOptions, setModelOptions] = useState<
    Array<{ id: string; name: string }>
  >([]);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const entriesRef = useRef<HistoryEntry[]>([]);
  const loadingRef = useRef(false);
  const requestGenerationRef = useRef(0);

  // Keep ref in sync for use in IntersectionObserver callback
  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  const loadPage = useCallback(
    async (cursor?: number) => {
      const isFirstPage = cursor === undefined;
      if (!isFirstPage && loadingRef.current) return;

      const generation = isFirstPage
        ? ++requestGenerationRef.current
        : requestGenerationRef.current;
      loadingRef.current = true;

      if (isFirstPage) setLoading(true);

      try {
        const result = await commands.getHistoryEntries(
          cursor ?? null,
          PAGE_SIZE,
          searchTerm || null,
          localMidnightTimestamp(startDate),
          localMidnightTimestamp(endDate, 1),
          modelFilter || null,
          outcomeFilter || null,
          cleanupFilter || null,
        );
        if (generation !== requestGenerationRef.current) {
          return;
        }
        if (result.status !== "ok") {
          throw new Error(String(result.error));
        }

        const { entries: newEntries, has_more } = result.data;
        setEntries((prev) =>
          isFirstPage ? newEntries : [...prev, ...newEntries],
        );
        setHasMore(has_more);
      } catch (error) {
        if (generation === requestGenerationRef.current) {
          console.error("Failed to load history entries:", error);
        }
      } finally {
        if (generation === requestGenerationRef.current) {
          setLoading(false);
          loadingRef.current = false;
        }
      }
    },
    [searchTerm, startDate, endDate, modelFilter, outcomeFilter, cleanupFilter],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearchTerm(searchQuery.trim());
    }, 200);
    return () => window.clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    void commands.getAvailableModels().then((result) => {
      if (result.status === "ok") {
        setModelOptions(
          result.data
            .map((model) => ({ id: model.id, name: model.name }))
            .sort((a, b) => a.name.localeCompare(b.name)),
        );
      }
    });
  }, []);

  // Initial load and full reload when the debounced search changes.
  useEffect(() => {
    loadPage();
  }, [loadPage]);

  // Infinite scroll via IntersectionObserver
  useEffect(() => {
    if (loading) return;

    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore) return;

    const observer = new IntersectionObserver(
      (observerEntries) => {
        const first = observerEntries[0];
        if (first.isIntersecting) {
          const lastEntry = entriesRef.current[entriesRef.current.length - 1];
          if (lastEntry) {
            loadPage(lastEntry.id);
          }
        }
      },
      { threshold: 0 },
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loading, hasMore, loadPage]);

  // Listen for new entries added from the transcription pipeline.
  useEffect(() => {
    const unlisten = events.historyUpdatePayload.listen((event) => {
      const payload: HistoryUpdatePayload = event.payload;
      if (payload.action === "added" || payload.action === "updated") {
        if (
          searchTerm ||
          startDate ||
          endDate ||
          modelFilter ||
          outcomeFilter ||
          cleanupFilter
        ) {
          // Re-run server-side filters so unrelated live updates never leak
          // into an active filtered result set.
          void loadPage();
        } else if (payload.action === "added") {
          setEntries((prev) => [payload.entry, ...prev]);
        } else {
          setEntries((prev) =>
            prev.map((e) => (e.id === payload.entry.id ? payload.entry : e)),
          );
        }
      }
      // "deleted" and "toggled" are handled by optimistic updates only,
      // so we intentionally ignore them here to avoid double-mutation.
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [
    loadPage,
    searchTerm,
    startDate,
    endDate,
    modelFilter,
    outcomeFilter,
    cleanupFilter,
  ]);

  const toggleSaved = async (id: number) => {
    // Optimistic update
    setEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
    );
    try {
      const result = await commands.toggleHistoryEntrySaved(id);
      if (result.status !== "ok") {
        // Revert on failure
        setEntries((prev) =>
          prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
        );
      }
    } catch (error) {
      console.error("Failed to toggle saved status:", error);
      // Revert on failure
      setEntries((prev) =>
        prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
      );
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      console.error("Failed to copy to clipboard:", error);
    }
  };

  const getAudioUrl = useCallback(
    async (fileName: string) => {
      try {
        const result = await commands.getAudioFilePath(fileName);
        if (result.status === "ok") {
          if (osType === "linux") {
            const fileData = await readFile(result.data);
            const blob = new Blob([fileData], { type: "audio/wav" });
            return URL.createObjectURL(blob);
          }
          return convertFileSrc(result.data, "asset");
        }
        return null;
      } catch (error) {
        console.error("Failed to get audio file path:", error);
        return null;
      }
    },
    [osType],
  );

  const deleteAudioEntry = async (id: number) => {
    // Optimistically remove
    setEntries((prev) => prev.filter((e) => e.id !== id));
    try {
      const result = await commands.deleteHistoryEntry(id);
      if (result.status !== "ok") {
        // Reload on failure
        loadPage();
      }
    } catch (error) {
      console.error("Failed to delete entry:", error);
      loadPage();
    }
  };

  const retryHistoryEntry = async (id: number) => {
    const result = await commands.retryHistoryEntryTranscription(id);
    if (result.status !== "ok") {
      throw new Error(String(result.error));
    }
  };

  const retryHistoryCleanup = async (id: number) => {
    const result = await commands.retryHistoryEntryCleanup(id);
    if (result.status !== "ok") {
      throw new Error(String(result.error));
    }
  };

  const openRecordingsFolder = async () => {
    try {
      const result = await commands.openRecordingsFolder();
      if (result.status !== "ok") {
        throw new Error(String(result.error));
      }
    } catch (error) {
      console.error("Failed to open recordings folder:", error);
    }
  };

  const hasActiveFilters = Boolean(
    searchTerm ||
      startDate ||
      endDate ||
      modelFilter ||
      outcomeFilter ||
      cleanupFilter,
  );

  let content: React.ReactNode;

  if (loading) {
    content = (
      <div className="px-4 py-3 text-center text-text/60">
        {t("settings.history.loading")}
      </div>
    );
  } else if (entries.length === 0) {
    content = (
      <div className="px-4 py-3 text-center text-text/60">
        {hasActiveFilters
          ? t("settings.history.noSearchResults", {
              defaultValue: "No matching history entries.",
            })
          : t("settings.history.empty")}
      </div>
    );
  } else {
    content = (
      <>
        <AudioPlayerGroup>
          <div className="divide-y divide-mid-gray/20">
            {entries.map((entry) => (
              <HistoryEntryComponent
                key={entry.id}
                entry={entry}
                onToggleSaved={() => toggleSaved(entry.id)}
                onCopyText={copyToClipboard}
                getAudioUrl={getAudioUrl}
                deleteAudio={deleteAudioEntry}
                retryTranscription={retryHistoryEntry}
                retryCleanup={retryHistoryCleanup}
              />
            ))}
          </div>
        </AudioPlayerGroup>
        {/* Sentinel for infinite scroll */}
        <div ref={sentinelRef} className="h-1" />
      </>
    );
  }

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="space-y-2">
        <div className="px-4 flex items-center justify-between gap-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
              {t("settings.history.title")}
            </h2>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              aria-label={t("settings.history.search", {
                defaultValue: "Search history",
              })}
              placeholder={t("settings.history.search", {
                defaultValue: "Search history...",
              })}
              className="mt-2 w-full rounded-md border border-mid-gray/20 bg-background px-3 py-1.5 text-sm text-text outline-none transition-colors placeholder:text-text/35 focus:border-logo-primary/60"
            />
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="text-[10px] font-medium uppercase tracking-wide text-text/45">
                {t("settings.history.fromDate", { defaultValue: "From" })}
                <input
                  type="date"
                  value={startDate}
                  max={endDate || undefined}
                  onChange={(event) => setStartDate(event.target.value)}
                  className="mt-1 block w-full rounded-md border border-mid-gray/20 bg-background px-2 py-1.5 text-sm font-normal normal-case tracking-normal text-text outline-none transition-colors focus:border-logo-primary/60"
                />
              </label>
              <label className="text-[10px] font-medium uppercase tracking-wide text-text/45">
                {t("settings.history.toDate", { defaultValue: "To" })}
                <input
                  type="date"
                  value={endDate}
                  min={startDate || undefined}
                  onChange={(event) => setEndDate(event.target.value)}
                  className="mt-1 block w-full rounded-md border border-mid-gray/20 bg-background px-2 py-1.5 text-sm font-normal normal-case tracking-normal text-text outline-none transition-colors focus:border-logo-primary/60"
                />
              </label>
            </div>
            <label className="mt-2 block text-[10px] font-medium uppercase tracking-wide text-text/45">
              {t("settings.history.modelFilter", { defaultValue: "Model" })}
              <select
                value={modelFilter}
                onChange={(event) => setModelFilter(event.target.value)}
                className="mt-1 block w-full rounded-md border border-mid-gray/20 bg-background px-2 py-1.5 text-sm font-normal normal-case tracking-normal text-text outline-none transition-colors focus:border-logo-primary/60"
              >
                <option value="">
                  {t("settings.history.allModels", { defaultValue: "All models" })}
                </option>
                {modelOptions.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="mt-2 block text-[10px] font-medium uppercase tracking-wide text-text/45">
              {t("settings.history.cleanupFilter", { defaultValue: "Cleanup" })}
              <select
                value={cleanupFilter}
                onChange={(event) => setCleanupFilter(event.target.value)}
                className="mt-1 block w-full rounded-md border border-mid-gray/20 bg-background px-2 py-1.5 text-sm font-normal normal-case tracking-normal text-text outline-none transition-colors focus:border-logo-primary/60"
              >
                <option value="">
                  {t("settings.history.allCleanupStates", {
                    defaultValue: "All cleanup states",
                  })}
                </option>
                <option value="requested">
                  {t("settings.history.cleanupRequested", {
                    defaultValue: "Cleanup requested",
                  })}
                </option>
                <option value="not_requested">
                  {t("settings.history.cleanupNotRequested", {
                    defaultValue: "No cleanup requested",
                  })}
                </option>
              </select>
            </label>
            <label className="mt-2 block text-[10px] font-medium uppercase tracking-wide text-text/45">
              {t("settings.history.outcomeFilter", { defaultValue: "Outcome" })}
              <select
                value={outcomeFilter}
                onChange={(event) => setOutcomeFilter(event.target.value)}
                className="mt-1 block w-full rounded-md border border-mid-gray/20 bg-background px-2 py-1.5 text-sm font-normal normal-case tracking-normal text-text outline-none transition-colors focus:border-logo-primary/60"
              >
                <option value="">
                  {t("settings.history.allOutcomes", { defaultValue: "All outcomes" })}
                </option>
                <option value="success">
                  {t("settings.history.successful", { defaultValue: "Successful" })}
                </option>
                <option value="failure">
                  {t("settings.history.failed", { defaultValue: "Failed" })}
                </option>
              </select>
            </label>
          </div>
          <OpenRecordingsButton
            onClick={openRecordingsFolder}
            label={t("settings.history.openFolder")}
          />
        </div>
        <div className="bg-background border border-mid-gray/20 rounded-lg overflow-visible">
          {content}
        </div>
      </div>
    </div>
  );
};

interface HistoryEntryProps {
  entry: HistoryEntry;
  onToggleSaved: () => void;
  onCopyText: (text: string) => void;
  getAudioUrl: (fileName: string) => Promise<string | null>;
  deleteAudio: (id: number) => Promise<void>;
  retryTranscription: (id: number) => Promise<void>;
  retryCleanup: (id: number) => Promise<void>;
}

const HistoryEntryComponent: React.FC<HistoryEntryProps> = ({
  entry,
  onToggleSaved,
  onCopyText,
  getAudioUrl,
  deleteAudio,
  retryTranscription,
  retryCleanup,
}) => {
  const { t, i18n } = useTranslation();
  const [showCopied, setShowCopied] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const busy = retrying || cleaning;

  const rawText = entry.transcription_text.trim();
  const finalText = entry.post_processed_text?.trim() ?? "";
  const hasTranscription = rawText.length > 0;
  const hasDistinctFinalText = finalText.length > 0 && finalText !== rawText;
  const copyText = finalText || rawText;

  const handleLoadAudio = useCallback(
    () => getAudioUrl(entry.file_name),
    [getAudioUrl, entry.file_name],
  );

  const handleCopyText = () => {
    if (!copyText) {
      return;
    }

    onCopyText(copyText);
    setShowCopied(true);
    setTimeout(() => setShowCopied(false), 2000);
  };

  const handleDeleteEntry = async () => {
    try {
      await deleteAudio(entry.id);
    } catch (error) {
      console.error("Failed to delete entry:", error);
      toast.error(t("settings.history.deleteError"));
    }
  };

  const handleRetranscribe = async () => {
    try {
      setRetrying(true);
      await retryTranscription(entry.id);
    } catch (error) {
      console.error("Failed to re-transcribe:", error);
      toast.error(t("settings.history.retranscribeError"));
    } finally {
      setRetrying(false);
    }
  };

  const handleRetryCleanup = async () => {
    try {
      setCleaning(true);
      await retryCleanup(entry.id);
    } catch (error) {
      console.error("Failed to retry cleanup:", error);
      toast.error(
        t("settings.history.retryCleanupError", {
          defaultValue: "Cleanup could not be retried.",
        }),
      );
    } finally {
      setCleaning(false);
    }
  };

  const formattedDate = formatDateTime(String(entry.timestamp), i18n.language);

  return (
    <div className="px-4 py-2 pb-5 flex flex-col gap-3">
      <div className="flex justify-between items-center">
        <p className="text-sm font-medium">{formattedDate}</p>
        <div className="flex items-center">
          <IconButton
            onClick={handleCopyText}
            disabled={!hasTranscription || busy}
            title={t("settings.history.copyToClipboard")}
          >
            {showCopied ? (
              <Check width={16} height={16} />
            ) : (
              <Copy width={16} height={16} />
            )}
          </IconButton>
          <IconButton
            onClick={onToggleSaved}
            disabled={busy}
            active={entry.saved}
            title={
              entry.saved
                ? t("settings.history.unsave")
                : t("settings.history.save")
            }
          >
            <Star
              width={16}
              height={16}
              fill={entry.saved ? "currentColor" : "none"}
            />
          </IconButton>
          <IconButton
            onClick={handleRetranscribe}
            disabled={busy}
            title={t("settings.history.retranscribe")}
          >
            <RotateCcw
              width={16}
              height={16}
              style={
                retrying
                  ? { animation: "spin 1s linear infinite reverse" }
                  : undefined
              }
            />
          </IconButton>
          <IconButton
            onClick={handleRetryCleanup}
            disabled={!hasTranscription || busy}
            title={t("settings.history.retryCleanup", {
              defaultValue: "Retry cleanup",
            })}
          >
            <Sparkles
              width={16}
              height={16}
              style={
                cleaning
                  ? { animation: "spin 1s linear infinite" }
                  : undefined
              }
            />
          </IconButton>
          <IconButton
            onClick={handleDeleteEntry}
            disabled={busy}
            title={t("settings.history.delete")}
          >
            <Trash2 width={16} height={16} />
          </IconButton>
        </div>
      </div>

      {(!hasDistinctFinalText || retrying) && (
        <p
          className={`italic text-sm pb-2 ${
            retrying
              ? ""
              : hasTranscription
                ? "text-text/90 select-text cursor-text whitespace-pre-wrap break-words"
                : "text-text/40"
          }`}
          style={
            retrying
              ? { animation: "transcribe-pulse 3s ease-in-out infinite" }
              : undefined
          }
        >
          {retrying && (
            <style>{`
              @keyframes transcribe-pulse {
                0%, 100% { color: color-mix(in srgb, var(--color-text) 40%, transparent); }
                50% { color: color-mix(in srgb, var(--color-text) 90%, transparent); }
              }
            `}</style>
          )}
          {retrying
            ? t("settings.history.transcribing")
            : hasTranscription
              ? rawText
              : t("settings.history.transcriptionFailed")}
        </p>
      )}

      {hasDistinctFinalText && !retrying && (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded-md border border-mid-gray/20 bg-mid-gray/5 p-3">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text/45">
              {t("settings.history.rawTranscript", { defaultValue: "Raw" })}
            </div>
            <p className="select-text whitespace-pre-wrap break-words text-sm text-text/75">
              {rawText}
            </p>
          </div>
          <div className="rounded-md border border-logo-primary/20 bg-logo-primary/5 p-3">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-logo-primary/80">
              {t("settings.history.finalTranscript", { defaultValue: "Final" })}
            </div>
            <p className="select-text whitespace-pre-wrap break-words text-sm text-text/90">
              {finalText}
            </p>
          </div>
        </div>
      )}

      <AudioPlayer onLoadRequest={handleLoadAudio} className="w-full" />
    </div>
  );
};
