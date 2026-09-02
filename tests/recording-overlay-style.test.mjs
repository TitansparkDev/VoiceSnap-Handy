import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readRepoFile = (path) =>
  readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("recording overlay keeps committed and tentative text in separate spans", async () => {
  const source = await readRepoFile("src/overlay/RecordingOverlay.tsx");

  const committedIndex = source.indexOf('className="committed"');
  const tentativeIndex = source.indexOf('className="tentative"');

  assert.notEqual(committedIndex, -1);
  assert.notEqual(tentativeIndex, -1);
  assert.ok(committedIndex < tentativeIndex);
});

test("recording overlay styles committed text more strongly than tentative text", async () => {
  const css = await readRepoFile("src/overlay/RecordingOverlay.css");

  assert.ok(
    css.includes(`.stext-cap .committed {
  color: var(--color-text);
  font-style: normal;
  font-weight: 500;
}`),
  );
  assert.ok(
    css.includes(`.stext-cap .tentative {
  color: var(--s-muted);
  font-style: italic;
  opacity: 0.72;
}`),
  );
});
