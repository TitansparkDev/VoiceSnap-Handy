import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import postcss from "postcss";

const overlayCss = await readFile(
  new URL("../src/overlay/RecordingOverlay.css", import.meta.url),
  "utf8",
);
const stylesheet = postcss.parse(overlayCss);

const declarationsFor = (selector: string) => {
  const declarations = new Map<string, string>();
  stylesheet.walkRules((rule) => {
    if (rule.selector !== selector) return;
    rule.walkDecls((declaration) => {
      declarations.set(declaration.prop, declaration.value);
    });
  });
  assert.ok(declarations.size > 0, `missing style rule for ${selector}`);
  return declarations;
};

const committed = declarationsFor(".stext-cap .committed");
const tentative = declarationsFor(".stext-cap .tentative");

assert.equal(committed.get("font-style"), "normal");
assert.equal(tentative.get("font-style"), "italic");
assert.ok(
  Number(committed.get("font-weight")) > Number(tentative.get("font-weight")),
  "committed transcript should carry more visual weight than tentative text",
);
assert.notEqual(
  committed.get("color"),
  tentative.get("color"),
  "committed and tentative transcript colors should be distinct",
);
