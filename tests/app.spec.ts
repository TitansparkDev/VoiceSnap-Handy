import { test, expect } from "@playwright/test";
import { getHistoryComparison } from "../src/components/settings/history/historyComparison";

test.describe("History comparison", () => {
  test("shows raw and post-processed text only when cleanup changed the transcript", () => {
    expect(
      getHistoryComparison({
        transcription_text: "raw words",
        post_processed_text: "clean words",
      }),
    ).toEqual({ raw: "raw words", final: "clean words" });

    expect(
      getHistoryComparison({
        transcription_text: "same words",
        post_processed_text: "  same words  ",
      }),
    ).toBeNull();
  });

  test("does not create a comparison for empty raw or cleanup output", () => {
    expect(
      getHistoryComparison({
        transcription_text: "raw words",
        post_processed_text: "   ",
      }),
    ).toBeNull();
    expect(
      getHistoryComparison({
        transcription_text: "   ",
        post_processed_text: "clean words",
      }),
    ).toBeNull();
  });
});

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
