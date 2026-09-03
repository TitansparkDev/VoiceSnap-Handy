import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  fixtureIdentity,
  parseArgs,
  parseFixture,
  percentile,
  summarizeSamples,
  WAVE2_MODELS,
} from "./stream-benchmark.mjs";

test("fixed fixture identity records basename and content hash without full path", () => {
  const dir = mkdtempSync(join(tmpdir(), "handy-stream-bench-"));
  const path = join(dir, "short.wav");
  writeFileSync(path, Buffer.from("fixed fixture bytes"));
  const identity = fixtureIdentity({ label: "short", path });
  assert.equal(identity.fixture, "short");
  assert.equal(identity.wav_file, "short.wav");
  assert.match(identity.wav_sha256, /^[0-9a-f]{64}$/);
  assert.equal(JSON.stringify(identity).includes(dir), false);
});

test("wave2 model shortcut expands the documented comparison set", () => {
  const options = parseArgs(
    ["--binary", "/tmp/handy", "--fixture", "short=/tmp/short.wav", "--wave2-models"],
    {},
  );
  assert.deepEqual(options.models, [...WAVE2_MODELS]);
});

test("fixture parser requires stable labels", () => {
  assert.equal(parseFixture("normal=./normal.wav").label, "normal");
  assert.throws(() => parseFixture("./normal.wav"), /Expected LABEL=WAV_PATH/);
});

test("final-only samples do not invent partial or cadence timing", () => {
  const summary = summarizeSamples([
    {
      mode: "final_only",
      first_partial_ms: null,
      committed_cadence_ms: [],
      finalization_tail_ms: 320,
      total_ms: 2320,
      worker_released: true,
    },
    {
      mode: "final_only",
      first_partial_ms: null,
      committed_cadence_ms: [],
      finalization_tail_ms: 280,
      total_ms: 2280,
      worker_released: true,
    },
  ]);
  assert.equal(summary.mode, "final_only");
  assert.deepEqual(summary.first_partial_ms, { p50: null, p95: null });
  assert.deepEqual(summary.committed_cadence_ms, { p50: null, p95: null, samples: 0 });
  assert.equal(summary.finalization_tail_ms.p50, 280);
});

test("streaming summary aggregates first partial, commit cadence, finalize and total timing", () => {
  const summary = summarizeSamples([
    {
      mode: "streaming",
      first_partial_ms: 140,
      committed_cadence_ms: [110, 120],
      finalization_tail_ms: 90,
      total_ms: 2100,
      worker_released: true,
    },
    {
      mode: "streaming",
      first_partial_ms: 180,
      committed_cadence_ms: [100, 130],
      finalization_tail_ms: 100,
      total_ms: 2200,
      worker_released: true,
    },
  ]);
  assert.equal(summary.first_partial_ms.p50, 140);
  assert.equal(summary.committed_cadence_ms.p50, 110);
  assert.equal(summary.finalization_tail_ms.p95, 100);
  assert.equal(summary.total_ms.p95, 2200);
  assert.equal(summary.worker_released, true);
});

test("nearest-rank percentile is deterministic", () => {
  assert.equal(percentile([4, 1, 3, 2], 50), 2);
  assert.equal(percentile([4, 1, 3, 2], 95), 4);
  assert.equal(percentile([], 50), null);
});
