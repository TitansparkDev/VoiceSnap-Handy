#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export const WAVE2_MODELS = Object.freeze([
  "handy-computer/parakeet-unified-en-0.6b-gguf",
  "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
  "handy-computer/moonshine-streaming-tiny-gguf",
  "handy-computer/whisper-large-v3-turbo-gguf",
]);

export function percentile(values, percent) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.ceil((percent / 100) * sorted.length) - 1);
  return sorted[index];
}

export function parseFixture(spec) {
  const separator = spec.indexOf("=");
  if (separator <= 0 || separator === spec.length - 1) {
    throw new Error(`Invalid --fixture '${spec}'. Expected LABEL=WAV_PATH.`);
  }
  return { label: spec.slice(0, separator).trim(), path: resolve(spec.slice(separator + 1).trim()) };
}

function parsePositiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(`${name} must be an integer >= 1.`);
  return parsed;
}

export function parseArgs(argv, env = process.env) {
  const options = {
    binary: env.HANDY_BENCHMARK_BINARY ? resolve(env.HANDY_BENCHMARK_BINARY) : null,
    fixtures: [],
    models: [],
    runs: 3,
    frameMs: 100,
    deviceIndex: null,
    jsonPath: null,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`${arg} requires a value.`);
      return argv[index];
    };
    switch (arg) {
      case "--binary": options.binary = resolve(value()); break;
      case "--fixture": options.fixtures.push(parseFixture(value())); break;
      case "--model": options.models.push(value()); break;
      case "--wave2-models": options.models.push(...WAVE2_MODELS); break;
      case "--runs": options.runs = parsePositiveInteger(value(), "--runs"); break;
      case "--frame-ms": options.frameMs = parsePositiveInteger(value(), "--frame-ms"); break;
      case "--device-index": options.deviceIndex = Number.parseInt(value(), 10); break;
      case "--json": options.jsonPath = value(); break;
      case "-h":
      case "--help": options.help = true; break;
      default: throw new Error(`Unknown argument '${arg}'. Use --help.`);
    }
  }
  options.models = [...new Set(options.models)];
  if (!options.help) {
    if (!options.binary) throw new Error("Pass --binary PATH or set HANDY_BENCHMARK_BINARY.");
    if (!options.fixtures.length) throw new Error("Pass at least one fixed --fixture LABEL=WAV_PATH.");
    if (!options.models.length) throw new Error("Pass --model ID or --wave2-models.");
    if (options.deviceIndex !== null && (!Number.isInteger(options.deviceIndex) || options.deviceIndex < 0)) {
      throw new Error("--device-index must be an integer >= 0.");
    }
  }
  return options;
}

function helpText() {
  return `VoiceSnap-Handy fixed-WAV streaming benchmark\n\nUsage:\n  npm run bench:stream -- --binary PATH --fixture short=short.wav --model MODEL_ID\n\nOptions:\n  --binary PATH             Built Handy executable. May also use HANDY_BENCHMARK_BINARY.\n  --fixture LABEL=WAV_PATH  Fixed 16 kHz mono 16-bit PCM fixture. Repeat for short/medium/long WAVs.\n  --model ID                Installed local catalog/custom model ID. Repeat to compare models.\n  --wave2-models            Compare Parakeet Unified, Nemotron Streaming 3.5, Moonshine Streaming Tiny, and Whisper Large v3 Turbo.\n  --runs N                  Repetitions per model/fixture with one cold load (default: 3).\n  --frame-ms N              Real-time feed frame size (default: 100 ms).\n  --device-index N          Optional transcribe-cpp device index from --list-devices.\n  --json PATH               Write safe aggregate JSON. Fixture/model text is never stored.\n  -h, --help                Show this help.\n\nThe harness hashes each WAV and records only its label, basename, SHA-256, duration, model identity, success/failure, worker-release state, and timing. Streaming runs report first partial, committed cadence, finalization tail, and total time. Final-only models intentionally report null first-partial/cadence metrics. Model assets must already be installed; this command performs no downloads.\n`;
}

export function fixtureIdentity(fixture) {
  if (!existsSync(fixture.path)) throw new Error(`Fixture '${fixture.label}' does not exist.`);
  const bytes = readFileSync(fixture.path);
  return {
    fixture: fixture.label,
    wav_file: basename(fixture.path),
    wav_sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function runProcess(binary, args) {
  return new Promise((resolvePromise) => {
    const child = spawn(binary, args, {
      stdio: ["ignore", "pipe", "ignore"],
      env: { ...process.env, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
    });
    let stdout = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      if (stdout.length < 512 * 1024) stdout += chunk;
    });
    child.on("error", () => resolvePromise({ code: -1, stdout: "" }));
    child.on("close", (code) => resolvePromise({ code: code ?? -1, stdout }));
  });
}

export function summarizeSamples(samples) {
  const modes = [...new Set(samples.map((sample) => sample.mode))];
  const firstPartial = samples.map((sample) => sample.first_partial_ms).filter(Number.isFinite);
  const cadence = samples.flatMap((sample) => sample.committed_cadence_ms ?? []).filter(Number.isFinite);
  const finalize = samples.map((sample) => sample.finalization_tail_ms).filter(Number.isFinite);
  const total = samples.map((sample) => sample.total_ms).filter(Number.isFinite);
  return {
    mode: modes.length === 1 ? modes[0] : "mixed",
    runs: samples.length,
    worker_released: samples.every((sample) => sample.worker_released === true),
    first_partial_ms: { p50: percentile(firstPartial, 50), p95: percentile(firstPartial, 95) },
    committed_cadence_ms: { p50: percentile(cadence, 50), p95: percentile(cadence, 95), samples: cadence.length },
    finalization_tail_ms: { p50: percentile(finalize, 50), p95: percentile(finalize, 95) },
    total_ms: { p50: percentile(total, 50), p95: percentile(total, 95) },
  };
}

export async function evaluate(options, fixture, model, processRunner = runProcess) {
  const identity = fixtureIdentity(fixture);
  const args = [
    "--transcribe-file", fixture.path,
    "--model", model,
    "--benchmark-stream",
    "--benchmark-frame-ms", String(options.frameMs),
    "--repeat", String(options.runs),
    "--json",
  ];
  if (options.deviceIndex !== null) args.push("--device-index", String(options.deviceIndex));
  const result = await processRunner(options.binary, args);
  if (result.code !== 0) {
    return { ...identity, model, success: false, error: "benchmark_process_failed" };
  }
  let parsed;
  try {
    parsed = JSON.parse(result.stdout.trim());
  } catch {
    return { ...identity, model, success: false, error: "invalid_benchmark_json" };
  }
  if (!Array.isArray(parsed.samples) || parsed.samples.length !== options.runs) {
    return { ...identity, model, success: false, error: "incomplete_benchmark_samples" };
  }
  const summary = summarizeSamples(parsed.samples);
  return {
    ...identity,
    model,
    success: summary.worker_released,
    audio_secs: parsed.audio_secs,
    load_ms: parsed.load_ms,
    bound_backend: parsed.bound_backend ?? null,
    ...summary,
  };
}

export async function evaluateBenchmarkMatrix(options, processRunner = runProcess) {
  const results = [];
  for (const fixture of options.fixtures) {
    for (const model of options.models) {
      results.push(await evaluate(options, fixture, model, processRunner));
    }
  }
  return results;
}

function printResults(results) {
  console.log("fixture\tmodel\tmode\tfirst_p50\tcadence_p50\tfinalize_p50\ttotal_p50\tworker_released\tsuccess");
  for (const result of results.results) {
    console.log([
      result.fixture,
      result.model,
      result.mode ?? "-",
      result.first_partial_ms?.p50 ?? "-",
      result.committed_cadence_ms?.p50 ?? "-",
      result.finalization_tail_ms?.p50 ?? "-",
      result.total_ms?.p50 ?? "-",
      result.worker_released ?? false,
      result.success ? "yes" : "no",
    ].join("\t"));
  }
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const options = parseArgs(argv, env);
  if (options.help) {
    console.log(helpText());
    return 0;
  }
  if (!existsSync(options.binary)) throw new Error(`Benchmark binary not found: ${options.binary}`);
  const results = {
    schema_version: 1,
    fixture_set: "fixed-wav-sha256",
    runs_per_fixture: options.runs,
    frame_ms: options.frameMs,
    results: await evaluateBenchmarkMatrix(options),
  };
  printResults(results);
  if (options.jsonPath) await writeFile(options.jsonPath, `${JSON.stringify(results, null, 2)}\n`, "utf8");
  return results.results.every((result) => result.success) ? 0 : 1;
}

const entry = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (import.meta.url === entry) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      console.error(`stream benchmark: ${error.message}`);
      process.exit(2);
    },
  );
}
