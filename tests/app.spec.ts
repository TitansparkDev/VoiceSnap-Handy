import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

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

  test("streaming transcript distinguishes committed and tentative text", () => {
    const css = readFileSync("src/overlay/RecordingOverlay.css", "utf8");
    const committedRule = css.match(
      /\.stext-cap \.committed\s*\{([^}]*)\}/,
    )?.[1];
    const tentativeRule = css.match(
      /\.stext-cap \.tentative\s*\{([^}]*)\}/,
    )?.[1];

    expect(committedRule).toContain("opacity: 1");
    expect(tentativeRule).toMatch(/opacity:\s*0\.\d+/);
    expect(tentativeRule).toContain("text-decoration-style: dotted");
  });
});
