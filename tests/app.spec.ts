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

  test("streaming overlay visually distinguishes committed and tentative text", async ({
    page,
  }) => {
    await page.setContent(`
      <div class="stext-cap">
        <span class="committed">Stable words</span>
        <span class="tentative">Draft words</span>
      </div>
    `);
    await page.addStyleTag({ path: "src/overlay/RecordingOverlay.css" });
    await page.evaluate(() => {
      document.documentElement.style.setProperty("--color-text", "#111111");
      document.documentElement.style.setProperty("--s-muted", "#777777");
    });

    const committed = page.locator(".committed");
    const tentative = page.locator(".tentative");

    await expect(committed).toHaveCSS("font-style", "normal");
    await expect(committed).toHaveCSS("font-weight", "500");
    await expect(tentative).toHaveCSS("font-style", "italic");
    await expect(tentative).toHaveCSS("opacity", "0.78");

    const committedColor = await committed.evaluate(
      (element) => getComputedStyle(element).color,
    );
    const tentativeColor = await tentative.evaluate(
      (element) => getComputedStyle(element).color,
    );
    expect(committedColor).not.toBe(tentativeColor);
  });
});
