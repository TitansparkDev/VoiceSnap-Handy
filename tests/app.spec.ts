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
    await page.goto("/src/overlay/index.html");

    await page.evaluate(() => {
      document.body.innerHTML = `
        <div class="stext-cap">
          <span class="committed">stable words</span>
          <span class="tentative">provisional words</span>
        </div>
      `;
    });

    const committed = page.locator(".committed");
    const tentative = page.locator(".tentative");

    await expect(committed).toHaveCSS("font-style", "normal");
    await expect(committed).toHaveCSS("font-weight", "500");
    await expect(tentative).toHaveCSS("font-style", "italic");
    await expect(tentative).toHaveCSS("opacity", "0.62");
    await expect(tentative).toHaveCSS("text-decoration-line", "underline");
    await expect(tentative).toHaveCSS("text-decoration-style", "dotted");
  });
});
