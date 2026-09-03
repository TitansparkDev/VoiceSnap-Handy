import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  evaluateBenchmarkMatrix,
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

test("fixed-WAV matrix exercises short medium long fixtures without inventing final-only partials", async () => {
  const dir = mkdtempSync(join(tmpdir(), "handy-stream-matrix-"));
  const fixtures = ["short", "medium", "long"].map((label, index) => {
    const path = join(dir, `${label}.wav`);
    writeFileSync(path, Buffer.alloc(index + 1, index + 1));
    return { label, path };
  });
  const options = {
    binary: "/fake/handy",
    fixtures,
    models: ["stream-model", "final-model"],
    runs: 2,
    frameMs: 80,
    deviceIndex: null,
    jsonPath: null,
    help: false,
  };
  const measured = { short: 140, medium: 420, long: 910 };
  const fakeRunner = async (_binary, args) => {
    const wav = args[args.indexOf("--transcribe-file") + 1];
    const model = args[args.indexOf("--model") + 1];
    const label = fixtures.find((fixture) => fixture.path === wav).label;
    const first = measured[label];
    const streaming = model === "stream-model";
    return {
      code: 0,
      stdout: JSON.stringify({
        audio_secs: { short: 1.2, medium: 8.5, long: 45.0 }[label],
        load_ms: 77,
        bound_backend: "cpu",
        samples: Array.from({ length: options.runs }, (_, run) => ({
          mode: streaming ? "streaming" : "final_only",
          first_partial_ms: streaming ? first + run : null,
          committed_cadence_ms: streaming ? [90 + run, 110 + run] : [],
          finalization_tail_ms: first + 30 + run,
          total_ms: first + 1000 + run,
          worker_released: true,
        })),
      }),
    };
  };

  const results = await evaluateBenchmarkMatrix(options, fakeRunner);
  assert.equal(results.length, 6);
  for (const label of ["short", "medium", "long"]) {
    const streaming = results.find((result) => result.fixture === label && result.model === "stream-model");
    const finalOnly = results.find((result) => result.fixture === label && result.model === "final-model");
    assert.equal(streaming.first_partial_ms.p50, measured[label]);
    assert.equal(streaming.committed_cadence_ms.p50, 91);
    assert.equal(streaming.committed_cadence_ms.samples, 4);
    assert.equal(streaming.finalization_tail_ms.p50, measured[label] + 30);
    assert.equal(streaming.total_ms.p50, measured[label] + 1000);
    assert.equal(streaming.worker_released, true);
    assert.deepEqual(finalOnly.first_partial_ms, { p50: null, p95: null });
    assert.deepEqual(finalOnly.committed_cadence_ms, { p50: null, p95: null, samples: 0 });
    assert.equal(finalOnly.mode, "final_only");
    assert.equal(finalOnly.success, true);
  }
});

test("nearest-rank percentile is deterministic", () => {
  assert.equal(percentile([4, 1, 3, 2], 50), 2);
  assert.equal(percentile([4, 1, 3, 2], 95), 4);
  assert.equal(percentile([], 50), null);
});
