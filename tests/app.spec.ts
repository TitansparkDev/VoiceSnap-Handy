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

  test("committed and tentative overlay text use distinct rendered styles", async ({
    page,
  }) => {
    await page.goto("/src/overlay/index.html");

    await page.evaluate(() => {
      document.body.innerHTML = `
        <div class="stext-cap">
          <p>
            <span class="committed">Committed words</span>
            <span class="tentative">Tentative words</span>
          </p>
        </div>
      `;
    });

    const committedStyle = await page.locator(".committed").evaluate((element) => {
      const style = getComputedStyle(element);
      return { color: style.color, opacity: style.opacity };
    });
    const tentativeStyle = await page.locator(".tentative").evaluate((element) => {
      const style = getComputedStyle(element);
      return { color: style.color, opacity: style.opacity };
    });

    expect(tentativeStyle.color).not.toBe(committedStyle.color);
    expect(Number(tentativeStyle.opacity)).toBeLessThan(
      Number(committedStyle.opacity),
    );
  });
});
