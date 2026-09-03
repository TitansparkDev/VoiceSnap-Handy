import { expect, test } from "@playwright/test";
import {
  formatRecommendedRuntime,
  getRuntimeRecoveryDiagnostic,
} from "../src/components/settings/history/runtimeDiagnostics";

const baseEntry = {
  backend: "cpu",
  device: "CPU",
  saved_accelerator: "auto",
  recommended_backend: "auto",
  recommended_device: null,
};

test("explains CPU recovery without changing saved accelerator intent", () => {
  expect(getRuntimeRecoveryDiagnostic(baseEntry)).toEqual({
    reason: "auto_cpu_fallback",
    recommendedLabel: "auto",
  });

  expect(
    getRuntimeRecoveryDiagnostic({
      ...baseEntry,
      saved_accelerator: "gpu",
      recommended_device: "Discrete GPU",
    }),
  ).toEqual({
    reason: "gpu_cpu_fallback",
    recommendedLabel: "auto · Discrete GPU",
  });
});

test("does not report recovery when CPU was requested or acceleration succeeded", () => {
  expect(
    getRuntimeRecoveryDiagnostic({
      ...baseEntry,
      saved_accelerator: "cpu",
      recommended_backend: "cpu",
    }),
  ).toBeNull();

  expect(
    getRuntimeRecoveryDiagnostic({
      ...baseEntry,
      backend: "vulkan",
      device: "Discrete GPU",
      saved_accelerator: "gpu",
      recommended_device: "Discrete GPU",
    }),
  ).toBeNull();
});

test("formats recommended backend and selected device as diagnostics", () => {
  expect(
    formatRecommendedRuntime({
      ...baseEntry,
      recommended_backend: "auto",
      recommended_device: "Discrete GPU",
    }),
  ).toBe("auto · Discrete GPU");

  expect(
    formatRecommendedRuntime({
      ...baseEntry,
      recommended_backend: null,
      recommended_device: null,
    }),
  ).toBeNull();
});
