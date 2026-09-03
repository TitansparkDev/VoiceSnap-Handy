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

test.describe("Recording overlay", () => {
  test("renders committed text distinctly from tentative text", async ({ page }) => {
    await installTauriEventMock(page);
    await page.goto("/src/overlay/index.html");

    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (window as unknown as TauriTestWindow).__tauriListenerCount(
              "show-overlay",
            ),
        ),
      )
      .toBeGreaterThan(0);

    await page.evaluate(() => {
      const testWindow = window as unknown as TauriTestWindow;
      testWindow.__emitTauriEvent("show-overlay", "streaming");
      testWindow.__emitTauriEvent("stream-text-event", {
        committed: "Stable committed words",
        tentative: "provisional suffix",
      });
    });

    const committed = page.locator(".stext-cap .committed");
    const tentative = page.locator(".stext-cap .tentative");

    await expect(committed).toHaveText("Stable committed words ");
    await expect(tentative).toHaveText("provisional suffix");

    const committedStyle = await committed.evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        color: style.color,
        fontStyle: style.fontStyle,
        fontWeight: style.fontWeight,
        opacity: style.opacity,
      };
    });
    const tentativeStyle = await tentative.evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        color: style.color,
        fontStyle: style.fontStyle,
        fontWeight: style.fontWeight,
        opacity: style.opacity,
      };
    });

    expect(committedStyle.color).not.toBe(tentativeStyle.color);
    expect(committedStyle.fontStyle).toBe("normal");
    expect(tentativeStyle.fontStyle).toBe("italic");
    expect(Number(tentativeStyle.opacity)).toBeLessThan(
      Number(committedStyle.opacity),
    );
    expect(Number(committedStyle.fontWeight)).toBeGreaterThan(
      Number(tentativeStyle.fontWeight),
    );
  });
});
