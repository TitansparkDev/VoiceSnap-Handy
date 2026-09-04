import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  evaluateBenchmarkMatrix,
  fixtureIdentity,
  inspectModelAvailability,
  parseArgs,
  parseFixture,
  parseReference,
  percentile,
  summarizeSamples,
  WAVE2_MODEL_CONTRACT,
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
    [
      "--binary",
      "/tmp/handy",
      "--fixture",
      "short=/tmp/short.wav",
      "--wave2-models",
    ],
    {},
  );
  assert.deepEqual(options.models, [...WAVE2_MODELS]);
});

test("fixture and reference parsers require stable labels", () => {
  assert.equal(parseFixture("normal=./normal.wav").label, "normal");
  assert.equal(parseReference("normal=./normal.txt").label, "normal");
  assert.throws(() => parseFixture("./normal.wav"), /Expected LABEL=WAV_PATH/);
  assert.throws(
    () => parseReference("./normal.txt"),
    /Expected LABEL=TEXT_PATH/,
  );
});

test("live microphone mode parses without requiring a physical fixture", () => {
  const options = parseArgs(
    [
      "--binary",
      "/tmp/handy",
      "--live-seconds",
      "8",
      "--live-reference",
      "/tmp/phrase.txt",
      "--model",
      "model",
    ],
    {},
  );
  assert.equal(options.liveSeconds, 8);
  assert.equal(options.fixtures.length, 0);
  assert.match(options.liveReference, /phrase\.txt$/);
});

test("Wave 2 catalog contract keeps streaming candidates and Whisper Turbo final-only", () => {
  const catalog = JSON.parse(
    readFileSync("src-tauri/src/catalog/catalog.json", "utf8"),
  );
  const byId = new Map(catalog.models.map((model) => [model.id, model]));
  for (const candidate of WAVE2_MODEL_CONTRACT) {
    const catalogModel = byId.get(candidate.catalog_id);
    assert.ok(catalogModel, `missing catalog model ${candidate.catalog_id}`);
    assert.equal(
      candidate.catalog_path,
      catalogModel.capabilities.streaming ? "streaming_capable" : "final_only",
    );
    const defaultFile = catalogModel.files.find(
      (file) => file.quant === catalogModel.default_quant,
    );
    assert.ok(defaultFile, `missing default quant for ${candidate.catalog_id}`);
    assert.equal(
      candidate.id,
      `${candidate.catalog_id}/${defaultFile.filename}`,
    );
  }
  const whisper = WAVE2_MODEL_CONTRACT.find((candidate) =>
    candidate.id.includes("whisper-large-v3-turbo"),
  );
  assert.equal(whisper.catalog_path, "final_only");
  assert.equal(
    WAVE2_MODEL_CONTRACT.filter(
      (candidate) => candidate.catalog_path === "streaming_capable",
    ).length,
    3,
  );
});

test("availability preflight distinguishes installed, missing, and unknown runtime IDs", async () => {
  const options = {
    binary: "/fake/handy",
    models: [WAVE2_MODELS[0], WAVE2_MODELS[1], "custom/missing"],
  };
  const fakeRunner = async (_binary, args) => {
    assert.deepEqual(args, ["--list-models", "--json"]);
    return {
      code: 0,
      stdout: JSON.stringify([
        {
          id: WAVE2_MODELS[0],
          is_downloaded: true,
          supports_streaming: true,
        },
        {
          id: WAVE2_MODELS[1],
          is_downloaded: false,
          supports_streaming: true,
        },
      ]),
    };
  };

  const availability = await inspectModelAvailability(options, fakeRunner);
  assert.deepEqual(availability.get(WAVE2_MODELS[0]), {
    availability: "installed",
    supports_streaming: true,
  });
  assert.deepEqual(availability.get(WAVE2_MODELS[1]), {
    availability: "not_installed",
    supports_streaming: true,
  });
  assert.deepEqual(availability.get("custom/missing"), {
    availability: "unknown_model",
    supports_streaming: null,
  });
});

test("matrix reports unavailable models with null timing and quality instead of launching them", async () => {
  const dir = mkdtempSync(join(tmpdir(), "handy-stream-unavailable-"));
  const wav = join(dir, "short.wav");
  writeFileSync(wav, Buffer.from("fixture"));
  const installed = WAVE2_MODELS[0];
  const unavailable = WAVE2_MODELS[1];
  const options = {
    binary: "/fake/handy",
    fixtures: [{ label: "short", path: wav }],
    references: [],
    models: [installed, unavailable],
    runs: 1,
    frameMs: 100,
    liveSeconds: null,
    deviceIndex: null,
  };
  const availability = new Map([
    [installed, { availability: "installed", supports_streaming: true }],
    [unavailable, { availability: "not_installed", supports_streaming: true }],
  ]);
  let benchmarkCalls = 0;
  const fakeRunner = async (_binary, args) => {
    benchmarkCalls += 1;
    assert.equal(args[args.indexOf("--model") + 1], installed);
    return {
      code: 0,
      stdout: JSON.stringify({
        audio_secs: 1,
        load_ms: 10,
        samples: [
          {
            mode: "streaming",
            first_partial_ms: 100,
            committed_cadence_ms: [80],
            finalization_tail_ms: 40,
            total_ms: 1040,
            worker_released: true,
            word_error_rate_milli: 0,
          },
        ],
      }),
    };
  };

  const results = await evaluateBenchmarkMatrix(
    options,
    fakeRunner,
    availability,
  );
  assert.equal(benchmarkCalls, 1);
  const missing = results.find((result) => result.model === unavailable);
  assert.equal(missing.success, false);
  assert.equal(missing.availability, "not_installed");
  assert.equal(missing.error, "model_not_installed");
  assert.deepEqual(missing.first_partial_ms, { p50: null, p95: null });
  assert.deepEqual(missing.committed_cadence_ms, {
    p50: null,
    p95: null,
    samples: 0,
  });
  assert.deepEqual(missing.quality, {
    metric: "word_error_rate_milli",
    p50: null,
    p95: null,
    samples: 0,
  });
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
  assert.deepEqual(summary.committed_cadence_ms, {
    p50: null,
    p95: null,
    samples: 0,
  });
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
      word_error_rate_milli: 125,
    },
    {
      mode: "streaming",
      first_partial_ms: 180,
      committed_cadence_ms: [100, 130],
      finalization_tail_ms: 100,
      total_ms: 2200,
      worker_released: true,
      word_error_rate_milli: 250,
    },
  ]);
  assert.equal(summary.first_partial_ms.p50, 140);
  assert.equal(summary.committed_cadence_ms.p50, 110);
  assert.equal(summary.finalization_tail_ms.p95, 100);
  assert.equal(summary.total_ms.p95, 2200);
  assert.deepEqual(summary.quality, {
    metric: "word_error_rate_milli",
    p50: 125,
    p95: 250,
    samples: 2,
  });
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
          word_error_rate_milli:
            { short: 0, medium: 125, long: 250 }[label] + run,
        })),
      }),
    };
  };

  const results = await evaluateBenchmarkMatrix(options, fakeRunner);
  assert.equal(results.length, 6);
  for (const label of ["short", "medium", "long"]) {
    const streaming = results.find(
      (result) => result.fixture === label && result.model === "stream-model",
    );
    const finalOnly = results.find(
      (result) => result.fixture === label && result.model === "final-model",
    );
    assert.equal(streaming.first_partial_ms.p50, measured[label]);
    assert.equal(streaming.committed_cadence_ms.p50, 91);
    assert.equal(streaming.committed_cadence_ms.samples, 4);
    assert.equal(streaming.finalization_tail_ms.p50, measured[label] + 30);
    assert.equal(streaming.total_ms.p50, measured[label] + 1000);
    assert.equal(streaming.quality.samples, 2);
    assert.equal(streaming.worker_released, true);
    assert.deepEqual(finalOnly.first_partial_ms, { p50: null, p95: null });
    assert.deepEqual(finalOnly.committed_cadence_ms, {
      p50: null,
      p95: null,
      samples: 0,
    });
    assert.equal(finalOnly.mode, "final_only");
    assert.equal(finalOnly.success, true);
  }
});

test("fixed-WAV reference is passed to the binary and reported by hash only", async () => {
  const dir = mkdtempSync(join(tmpdir(), "handy-stream-reference-"));
  const wav = join(dir, "short.wav");
  const reference = join(dir, "short.txt");
  writeFileSync(wav, Buffer.from("wav bytes"));
  writeFileSync(reference, "expected private transcript");
  const options = {
    binary: "/fake/handy",
    fixtures: [{ label: "short", path: wav }],
    references: [{ label: "short", path: reference }],
    models: ["model"],
    runs: 1,
    frameMs: 100,
    liveSeconds: null,
    deviceIndex: null,
  };
  const fakeRunner = async (_binary, args) => {
    assert.equal(
      args[args.indexOf("--benchmark-reference-file") + 1],
      reference,
    );
    return {
      code: 0,
      stdout: JSON.stringify({
        audio_secs: 1.5,
        load_ms: 20,
        samples: [
          {
            mode: "streaming",
            first_partial_ms: 120,
            committed_cadence_ms: [80],
            finalization_tail_ms: 50,
            total_ms: 1600,
            worker_released: true,
            word_error_rate_milli: 125,
          },
        ],
      }),
    };
  };
  const [result] = await evaluateBenchmarkMatrix(options, fakeRunner);
  assert.equal(result.quality.p50, 125);
  assert.match(result.reference_sha256, /^[0-9a-f]{64}$/);
  assert.equal(
    JSON.stringify(result).includes("expected private transcript"),
    false,
  );
  assert.equal(JSON.stringify(result).includes(dir), false);
});

test("live microphone benchmark uses the same timing and quality schema without hardware in CI", async () => {
  const dir = mkdtempSync(join(tmpdir(), "handy-stream-live-"));
  const reference = join(dir, "phrase.txt");
  writeFileSync(reference, "read this stable phrase");
  const options = {
    binary: "/fake/handy",
    fixtures: [],
    references: [],
    models: [WAVE2_MODELS[0], WAVE2_MODELS.at(-1)],
    runs: 2,
    frameMs: 100,
    liveSeconds: 6,
    liveLabel: "live-phrase",
    liveReference: reference,
    deviceIndex: null,
  };
  const fakeRunner = async (_binary, args) => {
    assert.equal(args.includes("--transcribe-file"), false);
    assert.equal(args[args.indexOf("--benchmark-live-seconds") + 1], "6");
    assert.equal(
      args[args.indexOf("--benchmark-reference-file") + 1],
      reference,
    );
    const model = args[args.indexOf("--model") + 1];
    const finalOnly = model.includes("whisper-large-v3-turbo");
    return {
      code: 0,
      stdout: JSON.stringify({
        input_mode: "live_microphone",
        audio_secs: 6,
        load_ms: 40,
        bound_backend: "cpu",
        samples: Array.from({ length: 2 }, (_, run) => ({
          mode: finalOnly ? "final_only" : "streaming",
          first_partial_ms: finalOnly ? null : 180 + run,
          committed_cadence_ms: finalOnly ? [] : [100 + run],
          finalization_tail_ms: 70 + run,
          total_ms: 6100 + run,
          worker_released: true,
          word_error_rate_milli: 100 + run,
        })),
      }),
    };
  };
  const results = await evaluateBenchmarkMatrix(options, fakeRunner);
  assert.equal(results.length, 2);
  const streaming = results.find(
    (result) => result.catalog_path === "streaming_capable",
  );
  const finalOnly = results.find(
    (result) => result.catalog_path === "final_only",
  );
  assert.equal(streaming.input_mode, "live_microphone");
  assert.equal(streaming.first_partial_ms.p50, 180);
  assert.equal(streaming.quality.p50, 100);
  assert.equal(finalOnly.mode, "final_only");
  assert.deepEqual(finalOnly.first_partial_ms, { p50: null, p95: null });
  assert.equal(
    JSON.stringify(results).includes("read this stable phrase"),
    false,
  );
});

test("nearest-rank percentile is deterministic", () => {
  assert.equal(percentile([4, 1, 3, 2], 50), 2);
  assert.equal(percentile([4, 1, 3, 2], 95), 4);
  assert.equal(percentile([], 50), null);
});
