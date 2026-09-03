import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useSettings } from "../../hooks/useSettings";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";
import type { VocabularyEntry } from "../../bindings";

interface CustomWordsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

const normalizeCustomWord = (word: string) =>
  word
    .replace(/[\u0000-\u001f\u007f<>"']/g, "")
    .replace(/\s+/g, " ")
    .trim();

export const CustomWords: React.FC<CustomWordsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const [newWord, setNewWord] = useState("");
    const [newAlias, setNewAlias] = useState("");
    const [newLanguage, setNewLanguage] = useState("");
    const vocabulary = getSetting("vocabulary_v1") || {
      version: 1,
      entries: [],
      replacements: [],
    };
    const entries = vocabulary.entries || [];
    const normalizedWord = normalizeCustomWord(newWord);
    const normalizedAlias = normalizeCustomWord(newAlias);
    const normalizedLanguage = normalizeCustomWord(newLanguage).replace(/_/g, "-");

    const updateEntries = (nextEntries: VocabularyEntry[]) =>
      updateSetting("vocabulary_v1", {
        ...vocabulary,
        version: 1,
        entries: nextEntries,
        replacements: vocabulary.replacements || [],
      });

    const handleAddWord = () => {
      if (normalizedWord && normalizedWord.length <= 80) {
        const duplicate = entries.some(
          (entry) =>
            entry.written.toLocaleLowerCase() ===
              normalizedWord.toLocaleLowerCase() &&
            (entry.spoken_alias || "").toLocaleLowerCase() ===
              normalizedAlias.toLocaleLowerCase() &&
            (entry.language || "").toLocaleLowerCase() ===
              normalizedLanguage.toLocaleLowerCase(),
        );
        if (duplicate) {
          toast.error(
            t("settings.advanced.customWords.duplicate", {
              word: normalizedWord,
            }),
          );
          return;
        }
        updateEntries([
          ...entries,
          {
            written: normalizedWord,
            spoken_alias: normalizedAlias || null,
            language: normalizedLanguage || null,
            enabled: true,
            case_sensitive: null,
            preserve_punctuation: null,
          },
        ]);
        setNewWord("");
        setNewAlias("");
        setNewLanguage("");
      }
    };

    const handleRemoveWord = (indexToRemove: number) => {
      updateEntries(entries.filter((_, index) => index !== indexToRemove));
    };

    const updateEntry = (index: number, patch: Partial<VocabularyEntry>) => {
      updateEntries(
        entries.map((entry, entryIndex) =>
          entryIndex === index ? { ...entry, ...patch } : entry,
        ),
      );
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddWord();
      }
    };

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.customWords.title")}
          description={t("settings.advanced.customWords.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <div className="flex flex-wrap items-center gap-2">
            <Input
              type="text"
              className="max-w-40"
              value={newWord}
              onChange={(e) => setNewWord(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.placeholder")}
              variant="compact"
              disabled={isUpdating("vocabulary_v1")}
            />
            <Input
              type="text"
              className="max-w-40"
              value={newAlias}
              onChange={(e) => setNewAlias(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.aliasPlaceholder", {
                defaultValue: "Spoken alias (optional)",
              })}
              variant="compact"
              disabled={isUpdating("vocabulary_v1")}
            />
            <Input
              type="text"
              className="max-w-28"
              value={newLanguage}
              onChange={(e) => setNewLanguage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.languagePlaceholder", {
                defaultValue: "Language (e.g. en)",
              })}
              variant="compact"
              disabled={isUpdating("vocabulary_v1")}
            />
            <Button
              onClick={handleAddWord}
              disabled={
                !normalizedWord ||
                normalizedWord.length > 80 ||
                normalizedAlias.length > 80 ||
                normalizedLanguage.length > 35 ||
                isUpdating("vocabulary_v1")
              }
              variant="primary"
              size="md"
            >
              {t("settings.advanced.customWords.add")}
            </Button>
          </div>
        </SettingContainer>
        {entries.length > 0 && (
          <div
            className={`px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"} flex flex-col gap-2`}
          >
            {entries.map((entry, index) => (
              <div
                key={`${entry.written}-${entry.spoken_alias || ""}-${entry.language || ""}-${index}`}
                className="flex flex-wrap items-center gap-2 text-sm"
              >
                <button
                  type="button"
                  className={`font-medium ${entry.enabled === false ? "opacity-50 line-through" : ""}`}
                  onClick={() =>
                    updateEntry(index, { enabled: entry.enabled === false })
                  }
                  disabled={isUpdating("vocabulary_v1")}
                  title={t("settings.advanced.customWords.toggleEnabled", {
                    defaultValue: "Enable or disable this vocabulary entry",
                  })}
                >
                  {entry.written}
                </button>
                {entry.spoken_alias && (
                  <span className="text-mid-gray">← {entry.spoken_alias}</span>
                )}
                {entry.language && (
                  <span className="rounded border border-mid-gray/20 px-1">
                    {entry.language}
                  </span>
                )}
                <label className="inline-flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    checked={entry.case_sensitive === true}
                    onChange={(event) =>
                      updateEntry(index, {
                        case_sensitive: event.target.checked ? true : null,
                      })
                    }
                    disabled={isUpdating("vocabulary_v1")}
                  />
                  {t("settings.advanced.customWords.caseSensitive", {
                    defaultValue: "Exact case",
                  })}
                </label>
                <label className="inline-flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    checked={entry.preserve_punctuation !== false}
                    onChange={(event) =>
                      updateEntry(index, {
                        preserve_punctuation: event.target.checked ? null : false,
                      })
                    }
                    disabled={isUpdating("vocabulary_v1")}
                  />
                  {t("settings.advanced.customWords.preservePunctuation", {
                    defaultValue: "Preserve punctuation",
                  })}
                </label>
                <Button
                  onClick={() => handleRemoveWord(index)}
                  disabled={isUpdating("vocabulary_v1")}
                  variant="secondary"
                  size="sm"
                  aria-label={t("settings.advanced.customWords.remove", {
                    word: entry.written,
                  })}
                >
                  ×
                </Button>
              </div>
            ))}
          </div>
        )}
      </>
    );
  },
);
