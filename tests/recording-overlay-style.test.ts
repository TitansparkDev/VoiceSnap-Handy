import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const overlayCss = readFileSync(
  new URL("../src/overlay/RecordingOverlay.css", import.meta.url),
  "utf8",
).replace('@import "../styles/theme.css";', "");

test("committed stream text is stronger than tentative text", async ({ page }) => {
  await page.setContent(`
    <style>
      :root {
        --color-text: rgb(20 20 20);
        --s-muted: rgb(120 120 120);
      }
    </style>
    <div class="stext-cap">
      <p>
        <span class="committed">Committed words</span>
        <span class="tentative">Tentative words</span>
      </p>
    </div>
  `);
  await page.addStyleTag({ content: overlayCss });

  const committed = page.locator(".committed");
  const tentative = page.locator(".tentative");

  await expect(committed).toHaveCSS("font-style", "normal");
  await expect(tentative).toHaveCSS("font-style", "italic");
  await expect(committed).toHaveCSS("font-weight", "500");
  await expect(tentative).toHaveCSS("font-weight", "400");

  const committedColor = await committed.evaluate(
    (element) => getComputedStyle(element).color,
  );
  const tentativeColor = await tentative.evaluate(
    (element) => getComputedStyle(element).color,
  );
  expect(tentativeColor).not.toBe(committedColor);
});