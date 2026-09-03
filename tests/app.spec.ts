import { test, expect } from "@playwright/test";

type TauriTestWindow = Window & {
  __emitTauriEvent: (event: string, payload: unknown) => void;
  __tauriListenerCount: (event: string) => number;
};

const installTauriEventMock = async (page: import("@playwright/test").Page) => {
  await page.addInitScript(() => {
    const callbacks = new Map<number, (payload: unknown) => void>();
    const listeners = new Map<string, number[]>();
    let nextCallbackId = 1;
    let nextEventId = 1;

    Object.assign(window, {
      __TAURI_INTERNALS__: {
        transformCallback(callback: (payload: unknown) => void) {
          const id = nextCallbackId++;
          callbacks.set(id, callback);
          return id;
        },
        unregisterCallback(id: number) {
          callbacks.delete(id);
        },
        async invoke(command: string, args: Record<string, unknown> = {}) {
          if (command === "plugin:event|listen") {
            const event = args.event as string;
            const handler = args.handler as number;
            listeners.set(event, [...(listeners.get(event) ?? []), handler]);
            return nextEventId++;
          }
          if (command === "plugin:event|unlisten") return null;
          if (command === "get_app_settings") {
            return {
              app_language: "en",
              overlay_position: "bottom",
              theme: "light",
            };
          }
          return null;
        },
      },
      __TAURI_EVENT_PLUGIN_INTERNALS__: {
        unregisterListener() {},
      },
      __emitTauriEvent(event: string, payload: unknown) {
        for (const callbackId of listeners.get(event) ?? []) {
          callbacks.get(callbackId)?.({ event, id: 0, payload });
        }
      },
      __tauriListenerCount(event: string) {
        return listeners.get(event)?.length ?? 0;
      },
    });
  });
};

test.describe("Handy App", () => {
  test("dev server responds", async ({ page }) => {
    // Just verify the dev server is running and responds
    const response = await page.goto("/");
    expect(response?.status()).toBe(200);
  });

  test("page has html structure", async ({ page }) => {
    await page.goto("/");

    // Verify basic HTML structure exists
    const html = await page.content();
    expect(html).toContain("<html");
    expect(html).toContain("<body");
  });
});
