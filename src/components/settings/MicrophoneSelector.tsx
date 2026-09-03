import React from "react";
import { useTranslation } from "react-i18next";
import { Dropdown } from "../ui/Dropdown";
import { SettingContainer } from "../ui/SettingContainer";
import { ResetButton } from "../ui/ResetButton";
import { useSettings } from "../../hooks/useSettings";
import type { AudioDevice } from "@/bindings";

const microphoneKey = (device: AudioDevice) => {
  if (device.is_default) return "default";
  if (device.stable_id) return `id:${device.stable_id}`;
  return `name:${device.name}:${device.index}`;
};

interface MicrophoneSelectorProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const MicrophoneSelector: React.FC<MicrophoneSelectorProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const {
      getSetting,
      resetSetting,
      isUpdating,
      isLoading,
      audioDevices,
      refreshAudioDevices,
      selectMicrophone,
    } = useSettings();

    const selectedMicrophoneName =
      getSetting("selected_microphone") === "default"
        ? "Default"
        : getSetting("selected_microphone") || "Default";
    const selectedMicrophoneId = getSetting("selected_microphone_id");
    const selectedDevice = audioDevices.find((device) =>
      selectedMicrophoneId
        ? device.stable_id === selectedMicrophoneId
        : selectedMicrophoneName === "Default"
          ? device.is_default
          : device.name === selectedMicrophoneName,
    );
    const selectedMicrophone = selectedDevice
      ? microphoneKey(selectedDevice)
      : null;

    const handleMicrophoneSelect = async (deviceKey: string) => {
      const device = audioDevices.find(
        (candidate) => microphoneKey(candidate) === deviceKey,
      );
      if (device) {
        await selectMicrophone(device);
      }
    };

    const handleReset = async () => {
      await resetSetting("selected_microphone");
    };

    const microphoneOptions = audioDevices.map((device) => ({
      value: microphoneKey(device),
      label: device.name,
    }));

    return (
      <SettingContainer
        title={t("settings.sound.microphone.title")}
        description={t("settings.sound.microphone.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <div className="flex items-center space-x-1">
          <Dropdown
            options={microphoneOptions}
            selectedValue={selectedMicrophone}
            onSelect={handleMicrophoneSelect}
            placeholder={
              isLoading || audioDevices.length === 0
                ? t("settings.sound.microphone.loading")
                : t("settings.sound.microphone.placeholder")
            }
            disabled={
              isUpdating("selected_microphone") ||
              isLoading ||
              audioDevices.length === 0
            }
            onRefresh={refreshAudioDevices}
          />
          <ResetButton
            onClick={handleReset}
            disabled={isUpdating("selected_microphone") || isLoading}
          />
        </div>
      </SettingContainer>
    );
  },
);
