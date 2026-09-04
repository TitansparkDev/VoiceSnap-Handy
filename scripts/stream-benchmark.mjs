#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export const WAVE2_MODEL_CONTRACT = Object.freeze([
  {
    id: "handy-computer/parakeet-unified-en-0.6b-gguf/parakeet-unified-en-0.6b-Q8_0.gguf",
    catalog_id: "handy-computer/parakeet-unified-en-0.6b-gguf",
    catalog_path: "streaming_capable",
  },
  {
    id: "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
    catalog_id: "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
    catalog_path: "streaming_capable",
  },
  {
    id: "handy-computer/moonshine-streaming-tiny-gguf/moonshine-streaming-tiny-Q8_0.gguf",
    catalog_id: "handy-computer/moonshine-streaming-tiny-gguf",
    catalog_path: "streaming_capable",
  },
  {
    id: "handy-computer/whisper-large-v3-turbo-gguf/whisper-large-v3-turbo-Q8_0.gguf",
    catalog_id: "handy-computer/whisper-large-v3-turbo-gguf",
    catalog_path: "final_only",
  },
]);

export const WAVE2_MODELS = Object.freeze(
  WAVE2_MODEL_CONTRACT.map((candidate) => candidate.id),
);

export function percentile(values, percent) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.ceil((percent / 100) * sorted.length) - 1);
  return sorted[index];
}

function parseLabeledPath(spec, flag, expected) {
  const separator = spec.indexOf("=");
  if (separator <= 0 || separator === spec.length - 1) {
    throw new Error(`Invalid ${flag} '${spec}'. Expected ${expected}.`);
  }
  return {
    label: spec.slice(0, separator).trim(),
    path: resolve(spec.slice(separator + 1).trim()),
  };
}

export function parseFixture(spec) {
  return parseLabeledPath(spec, "--fixture", "LABEL=WAV_PATH");
}

export function parseReference(spec) {
  return parseLabeledPath(spec, "--reference", "LABEL=TEXT_PATH");
}

function parsePositiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 1)
    throw new Error(`${name} must be an integer >= 1.`);
  return parsed;
}

export function parseArgs(argv, env = process.env) {
  const options = {
    binary: env.HANDY_BENCHMARK_BINARY
      ? resolve(env.HANDY_BENCHMARK_BINARY)
      : null,
    fixtures: [],
    references: [],
    models: [],
    runs: 3,
    frameMs: 100,
    liveSeconds: null,
    liveLabel: "live",
    liveReference: null,
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
      case "--binary":
        options.binary = resolve(value());
        break;
      case "--fixture":
        options.fixtures.push(parseFixture(value()));
        break;
      case "--reference":
        options.references.push(parseReference(value()));
        break;
      case "--live-seconds":
        options.liveSeconds = parsePositiveInteger(value(), "--live-seconds");
        break;
      case "--live-label":
        options.liveLabel = value();
        break;
      case "--live-reference":
        options.liveReference = resolve(value());
        break;
      case "--model":
        options.models.push(value());
        break;
      case "--wave2-models":
        options.models.push(...WAVE2_MODELS);
        break;
      case "--runs":
        options.runs = parsePositiveInteger(value(), "--runs");
        break;
      case "--frame-ms":
        options.frameMs = parsePositiveInteger(value(), "--frame-ms");
        break;
      case "--device-index":
        options.deviceIndex = Number.parseInt(value(), 10);
        break;
      case "--json":
        options.jsonPath = value();
        break;
      case "-h":
      case "--help":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown argument '${arg}'. Use --help.`);
    }
  }
  options.models = [...new Set(options.models)];
  if (!options.help) {
    if (!options.binary)
      throw new Error("Pass --binary PATH or set HANDY_BENCHMARK_BINARY.");
    if (!options.fixtures.length && options.liveSeconds === null) {
      throw new Error(
        "Pass at least one fixed --fixture LABEL=WAV_PATH or --live-seconds N.",
      );
    }
    if (!options.models.length)
      throw new Error("Pass --model ID or --wave2-models.");
    if (
      options.deviceIndex !== null &&
      (!Number.isInteger(options.deviceIndex) || options.deviceIndex < 0)
    ) {
      throw new Error("--device-index must be an integer >= 0.");
    }
  }
  return options;
}

function helpText() {
  return `VoiceSnap-Handy stream benchmark\n\nUsage:\n  npm run bench:stream -- --binary PATH --fixture short=short.wav --reference short=short.txt --model MODEL_ID\n  npm run bench:stream -- --binary PATH --live-seconds 10 --live-reference phrase.txt --model MODEL_ID\n\nOptions:\n  --binary PATH               Built Handy executable. May also use HANDY_BENCHMARK_BINARY.\n  --fixture LABEL=WAV_PATH    Fixed 16 kHz mono 16-bit PCM fixture. Repeat for short/medium/long WAVs.\n  --reference LABEL=TEXT_PATH Optional reference transcript for deterministic WER; label must match a fixture.\n  --live-seconds N            Record a real microphone session for N seconds through the same StreamRouter.\n  --live-label LABEL          Stable aggregate label for live sessions (default: live).\n  --live-reference TEXT_PATH  Optional phrase reference for deterministic live-session WER.\n  --model ID                  Installed local catalog/custom model ID. Repeat to compare models.\n  --wave2-models              Compare the default Q8 catalog entries for Parakeet Unified, Nemotron Streaming 3.5, Moonshine Streaming Tiny, and Whisper Large v3 Turbo.\n  --runs N                    Repetitions per model/input with one cold load (default: 3).\n  --frame-ms N                Fixed-WAV real-time feed frame size (default: 100 ms).\n  --device-index N            Optional transcribe-cpp device index from --list-devices.\n  --json PATH                 Write safe aggregate JSON. Transcript/reference text is never stored.\n  -h, --help                  Show this help.\n\nFixed WAV and live microphone modes use the same timing schema. Optional references add word-error-rate in thousandths (0 = exact, 1000 = 100%) without emitting recognized or expected text. Streaming runs report first partial, committed cadence, finalization tail, and total time. Final-only models intentionally report null first-partial/cadence metrics. Model assets must already be installed; this command performs no downloads.\n`;
}

function safeFileIdentity(path, kind) {
  if (!existsSync(path)) throw new Error(`${kind} does not exist.`);
  const bytes = readFileSync(path);
  return {
    file: basename(path),
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

export function fixtureIdentity(fixture) {
  const identity = safeFileIdentity(fixture.path, `Fixture '${fixture.label}'`);
  return {
    fixture: fixture.label,
    wav_file: identity.file,
    wav_sha256: identity.sha256,
  };
}

function referenceIdentity(path) {
  if (!path) return {};
  const identity = safeFileIdentity(path, "Benchmark reference");
  return { reference_file: identity.file, reference_sha256: identity.sha256 };
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

export async function inspectModelAvailability(
  options,
  processRunner = runProcess,
) {
  const result = await processRunner(options.binary, [
    "--list-models",
    "--json",
  ]);
  if (result.code !== 0) return null;

  let parsed;
  try {
    parsed = JSON.parse(result.stdout.trim());
  } catch {
    return null;
  }
  if (!Array.isArray(parsed)) return null;

  const byId = new Map(parsed.map((model) => [model.id, model]));
  return new Map(
    options.models.map((model) => {
      const listed = byId.get(model);
      if (!listed) {
        return [
          model,
          { availability: "unknown_model", supports_streaming: null },
        ];
      }
      return [
        model,
        {
          availability: listed.is_downloaded ? "installed" : "not_installed",
          supports_streaming:
            typeof listed.supports_streaming === "boolean"
              ? listed.supports_streaming
              : null,
        },
      ];
    }),
  );
}

export function summarizeSamples(samples) {
  const modes = [...new Set(samples.map((sample) => sample.mode))];
  const firstPartial = samples
    .map((sample) => sample.first_partial_ms)
    .filter(Number.isFinite);
  const cadence = samples
    .flatMap((sample) => sample.committed_cadence_ms ?? [])
    .filter(Number.isFinite);
  const finalize = samples
    .map((sample) => sample.finalization_tail_ms)
    .filter(Number.isFinite);
  const total = samples
    .map((sample) => sample.total_ms)
    .filter(Number.isFinite);
  const quality = samples
    .map((sample) => sample.word_error_rate_milli)
    .filter(Number.isFinite);
  return {
    mode: modes.length === 1 ? modes[0] : "mixed",
    runs: samples.length,
    worker_released: samples.every((sample) => sample.worker_released === true),
    first_partial_ms: {
      p50: percentile(firstPartial, 50),
      p95: percentile(firstPartial, 95),
    },
    committed_cadence_ms: {
      p50: percentile(cadence, 50),
      p95: percentile(cadence, 95),
      samples: cadence.length,
    },
    finalization_tail_ms: {
      p50: percentile(finalize, 50),
      p95: percentile(finalize, 95),
    },
    total_ms: { p50: percentile(total, 50), p95: percentile(total, 95) },
    quality: {
      metric: "word_error_rate_milli",
      p50: percentile(quality, 50),
      p95: percentile(quality, 95),
      samples: quality.length,
    },
  };
}

export async function evaluate(
  options,
  fixture,
  model,
  processRunner = runProcess,
) {
  const identity = fixtureIdentity(fixture);
  const reference = (options.references ?? []).find(
    (candidate) => candidate.label === fixture.label,
  );
  const args = [
    "--transcribe-file",
    fixture.path,
    "--model",
    model,
    "--benchmark-stream",
    "--benchmark-frame-ms",
    String(options.frameMs),
    "--repeat",
    String(options.runs),
    "--json",
  ];
  if (reference) args.push("--benchmark-reference-file", reference.path);
  if (options.deviceIndex !== null)
    args.push("--device-index", String(options.deviceIndex));
  const result = await processRunner(options.binary, args);
  if (result.code !== 0) {
    return {
      ...identity,
      model,
      success: false,
      error: "benchmark_process_failed",
    };
  }
  let parsed;
  try {
    parsed = JSON.parse(result.stdout.trim());
  } catch {
    return {
      ...identity,
      model,
      success: false,
      error: "invalid_benchmark_json",
    };
  }
  if (
    !Array.isArray(parsed.samples) ||
    parsed.samples.length !== options.runs
  ) {
    return {
      ...identity,
      model,
      success: false,
      error: "incomplete_benchmark_samples",
    };
  }
  const summary = summarizeSamples(parsed.samples);
  const contract = WAVE2_MODEL_CONTRACT.find(
    (candidate) => candidate.id === model,
  );
  return {
    ...identity,
    ...referenceIdentity(reference?.path),
    input_mode: "fixed_wav",
    model,
    catalog_path: contract?.catalog_path ?? "custom_or_unclassified",
    availability: "installed",
    success: summary.worker_released,
    audio_secs: parsed.audio_secs,
    load_ms: parsed.load_ms,
    bound_backend: parsed.bound_backend ?? null,
    ...summary,
  };
}

export async function evaluateLive(options, model, processRunner = runProcess) {
  const args = [
    "--model",
    model,
    "--benchmark-live-seconds",
    String(options.liveSeconds),
    "--repeat",
    String(options.runs),
    "--json",
  ];
  if (options.liveReference)
    args.push("--benchmark-reference-file", options.liveReference);
  if (options.deviceIndex !== null)
    args.push("--device-index", String(options.deviceIndex));
  const result = await processRunner(options.binary, args);
  const identity = {
    fixture: options.liveLabel,
    ...referenceIdentity(options.liveReference),
  };
  if (result.code !== 0) {
    return {
      ...identity,
      input_mode: "live_microphone",
      model,
      success: false,
      error: "benchmark_process_failed",
    };
  }
  let parsed;
  try {
    parsed = JSON.parse(result.stdout.trim());
  } catch {
    return {
      ...identity,
      input_mode: "live_microphone",
      model,
      success: false,
      error: "invalid_benchmark_json",
    };
  }
  if (
    !Array.isArray(parsed.samples) ||
    parsed.samples.length !== options.runs
  ) {
    return {
      ...identity,
      input_mode: "live_microphone",
      model,
      success: false,
      error: "incomplete_benchmark_samples",
    };
  }
  const summary = summarizeSamples(parsed.samples);
  const contract = WAVE2_MODEL_CONTRACT.find(
    (candidate) => candidate.id === model,
  );
  return {
    ...identity,
    input_mode: "live_microphone",
    live_seconds: options.liveSeconds,
    model,
    catalog_path: contract?.catalog_path ?? "custom_or_unclassified",
    availability: "installed",
    success: summary.worker_released,
    audio_secs: parsed.audio_secs,
    load_ms: parsed.load_ms,
    bound_backend: parsed.bound_backend ?? null,
    ...summary,
  };
}

function unavailableBenchmarkResult(
  options,
  fixture,
  model,
  status,
  inputMode,
) {
  const contract = WAVE2_MODEL_CONTRACT.find(
    (candidate) => candidate.id === model,
  );
  const reference =
    inputMode === "fixed_wav"
      ? (options.references ?? []).find(
          (candidate) => candidate.label === fixture.label,
        )
      : null;
  const identity =
    inputMode === "fixed_wav"
      ? { ...fixtureIdentity(fixture), ...referenceIdentity(reference?.path) }
      : {
          fixture: options.liveLabel,
          ...referenceIdentity(options.liveReference),
        };
  const error =
    status.availability === "not_installed"
      ? "model_not_installed"
      : "model_not_listed";
  return {
    ...identity,
    input_mode: inputMode,
    model,
    catalog_path: contract?.catalog_path ?? "custom_or_unclassified",
    availability: status.availability,
    success: false,
    error,
    mode: null,
    worker_released: false,
    first_partial_ms: { p50: null, p95: null },
    committed_cadence_ms: { p50: null, p95: null, samples: 0 },
    finalization_tail_ms: { p50: null, p95: null },
    total_ms: { p50: null, p95: null },
    quality: {
      metric: "word_error_rate_milli",
      p50: null,
      p95: null,
      samples: 0,
    },
  };
}

export async function evaluateBenchmarkMatrix(
  options,
  processRunner = runProcess,
  availabilityByModel = null,
) {
  const results = [];
  for (const fixture of options.fixtures) {
    for (const model of options.models) {
      const status = availabilityByModel?.get(model);
      if (status && status.availability !== "installed") {
        results.push(
          unavailableBenchmarkResult(
            options,
            fixture,
            model,
            status,
            "fixed_wav",
          ),
        );
        continue;
      }
      results.push(await evaluate(options, fixture, model, processRunner));
    }
  }
  if (options.liveSeconds !== null && options.liveSeconds !== undefined) {
    for (const model of options.models) {
      const status = availabilityByModel?.get(model);
      if (status && status.availability !== "installed") {
        results.push(
          unavailableBenchmarkResult(
            options,
            null,
            model,
            status,
            "live_microphone",
          ),
        );
        continue;
      }
      results.push(await evaluateLive(options, model, processRunner));
    }
  }
  return results;
}

function printResults(results) {
  console.log(
    "input\tfixture\tmodel\tavailability\tmode\tquality_wer_milli_p50\tfirst_p50\tcadence_p50\tfinalize_p50\ttotal_p50\tworker_released\tsuccess",
  );
  for (const result of results.results) {
    console.log(
      [
        result.input_mode ?? "-",
        result.fixture,
        result.model,
        result.availability ?? "unchecked",
        result.mode ?? "-",
        result.quality?.p50 ?? "-",
        result.first_partial_ms?.p50 ?? "-",
        result.committed_cadence_ms?.p50 ?? "-",
        result.finalization_tail_ms?.p50 ?? "-",
        result.total_ms?.p50 ?? "-",
        result.worker_released ?? false,
        result.success ? "yes" : "no",
      ].join("\t"),
    );
  }
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const options = parseArgs(argv, env);
  if (options.help) {
    console.log(helpText());
    return 0;
  }
  if (!existsSync(options.binary))
    throw new Error(`Benchmark binary not found: ${options.binary}`);
  const availability = await inspectModelAvailability(options);
  const results = {
    schema_version: 3,
    fixture_set:
      options.liveSeconds === null
        ? "fixed-wav-sha256"
        : "fixed-wav-and-live-microphone",
    runs_per_fixture: options.runs,
    frame_ms: options.frameMs,
    availability_checked: availability !== null,
    results: await evaluateBenchmarkMatrix(options, runProcess, availability),
  };
  printResults(results);
  if (options.jsonPath)
    await writeFile(
      options.jsonPath,
      `${JSON.stringify(results, null, 2)}\n`,
      "utf8",
    );
  return results.results.every((result) => result.success) ? 0 : 1;
}

const entry = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (import.meta.url === entry) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      console.error(`stream benchmark: ${error.message}`);
      process.exit(2);
    },
  );
}
