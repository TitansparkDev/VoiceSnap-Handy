import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { InsertionMode } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Alert } from "@/components/ui/Alert";
import { Dropdown, type DropdownOption } from "@/components/ui/Dropdown";
import { SettingContainer } from "@/components/ui/SettingContainer";

interface InsertionModeSettingProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const InsertionModeSetting: React.FC<InsertionModeSettingProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const selectedMode = getSetting("insertion_mode") ?? "at_stop";

  const options = useMemo<DropdownOption[]>(
    () => [
      {
        value: "at_stop",
        label: t("settings.advanced.insertionMode.options.atStop"),
      },
      {
        value: "preview_only",
        label: t("settings.advanced.insertionMode.options.previewOnly"),
      },
      {
        value: "live_committed_experimental",
        label: t(
          "settings.advanced.insertionMode.options.liveCommittedExperimental",
        ),
      },
    ],
    [t],
  );

  return (
    <>
      <SettingContainer
        title={t("settings.advanced.insertionMode.title")}
        description={t("settings.advanced.insertionMode.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        layout="horizontal"
      >
        <Dropdown
          options={options}
          selectedValue={selectedMode}
          onSelect={(value) =>
            updateSetting("insertion_mode", value as InsertionMode)
          }
          disabled={isUpdating("insertion_mode")}
        />
      </SettingContainer>
      {selectedMode === "live_committed_experimental" && (
        <Alert variant="warning" contained={grouped}>
          {t("settings.advanced.insertionMode.liveWarning")}
        </Alert>
      )}
    </>
  );
};
