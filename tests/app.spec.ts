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

  test("committed and tentative overlay text render distinctly", async ({ page }) => {
    await page.setContent(`
      <link rel="stylesheet" href="http://localhost:1420/src/overlay/RecordingOverlay.css" />
      <div class="stext-cap">
        <p>
          <span class="committed">Stable transcript</span>
          <span class="tentative">Provisional transcript</span>
        </p>
      </div>
    `);

    const committed = page.locator(".committed");
    const tentative = page.locator(".tentative");

    await expect(committed).toHaveCSS("font-style", "normal");
    await expect(tentative).toHaveCSS("font-style", "italic");
    await expect(tentative).toHaveCSS("opacity", "0.72");
  });
});
