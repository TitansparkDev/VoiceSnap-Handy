import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import {
  assertLoopbackEndpoint,
  candidateIdentity,
  completionBody,
  extractModelPath,
  normalizeCleanupOutput,
  offlineRuntimeEnv,
  parseArgs,
  parseCandidateProfileSpec,
  parseCandidateSpec,
  percentile,
  replaceModelPath,
  runFixture,
  SHORT_DICTATION_FIXTURE_ID,
  summarizeCandidate,
} from "./cleanup-benchmark.mjs";

test("candidate specs separate reporting labels from local asset paths", () => {
  const candidate = parseCandidateSpec("qwen3-1.7b=./models/qwen.gguf");
  assert.equal(candidate.label, "qwen3-1.7b");
  assert.equal(candidate.modelPath.endsWith("models/qwen.gguf"), true);
  assert.deepEqual(candidateIdentity(candidate), {
    candidate: "qwen3-1.7b",
    model_asset: "qwen.gguf",
    prompt_profile: "generic-v1",
  });
  assert.equal(JSON.stringify(candidateIdentity(candidate)).includes(candidate.modelPath), false);
});

test("candidate prompt profiles are explicit and validated", () => {
  assert.deepEqual(parseCandidateProfileSpec("s1-mini=s1-mini-v1"), {
    label: "s1-mini",
    profile: "s1-mini-v1",
  });
  assert.throws(() => parseCandidateProfileSpec("s1-mini=unknown"), /Unsupported cleanup prompt profile/);

  const options = parseArgs(
    [
      "--candidate",
      "s1-mini=./s1.gguf",
      "--candidate-profile",
      "s1-mini=s1-mini-v1",
    ],
    {},
  );
  assert.equal(options.candidates[0].profile, "s1-mini-v1");
  assert.throws(
    () => parseArgs(["--candidate", "qwen=./q.gguf", "--candidate-profile", "missing=s1-mini-v1"], {}),
    /unknown candidate/,
  );
});

test("configured model path is extracted from common llama.cpp argument forms", () => {
  assert.equal(extractModelPath(["--host", "127.0.0.1", "-m", "./a.gguf"]).endsWith("a.gguf"), true);
  assert.equal(extractModelPath(["--model=./b.gguf"]).endsWith("b.gguf"), true);
  assert.equal(extractModelPath(["--model-file", "./c.gguf"]).endsWith("c.gguf"), true);
  assert.equal(extractModelPath(["--port", "8080"]), null);
});

test("candidate replacement preserves the runtime argument template", () => {
  assert.deepEqual(replaceModelPath(["--port", "8080", "-m", "old.gguf"], "/models/new.gguf"), [
    "--port",
    "8080",
    "-m",
    "/models/new.gguf",
  ]);
  assert.deepEqual(replaceModelPath(["--port", "8080"], "/models/new.gguf", "--model"), [
    "--port",
    "8080",
    "--model",
    "/models/new.gguf",
  ]);
});

test("no explicit candidate defaults to the current configured local model", () => {
  const options = parseArgs([], {
    HANDY_LOCAL_CLEANUP_COMMAND: "llama-server",
    HANDY_LOCAL_CLEANUP_ARGS: JSON.stringify(["--port", "8080", "-m", "./configured.gguf"]),
  });
  assert.equal(options.candidates.length, 1);
  assert.equal(options.candidates[0].label, "configured");
  assert.equal(options.candidates[0].modelPath.endsWith("configured.gguf"), true);
});

test("benchmark refuses non-loopback endpoints", () => {
  assert.equal(assertLoopbackEndpoint("http://localhost:8080/v1"), "http://localhost:8080/v1");
  assert.throws(() => assertLoopbackEndpoint("https://example.com/v1"), /loopback-only/);
  assert.throws(() => assertLoopbackEndpoint("http://192.168.1.20:8080/v1"), /loopback-only/);
});

test("benchmark child runtime is explicitly offline and proxy-independent", () => {
  const env = offlineRuntimeEnv({ HTTPS_PROXY: "http://proxy.example:3128" });
  assert.equal(env.HF_HUB_OFFLINE, "1");
  assert.equal(env.TRANSFORMERS_OFFLINE, "1");
  assert.equal(env.HF_HUB_DISABLE_TELEMETRY, "1");
  assert.equal(env.NO_PROXY, "*");
  assert.equal(env.no_proxy, "*");
});

test("cleanup model request contains transcript text only, never ambient application data", () => {
  const fixture = { id: "privacy", input: "only this transcript may be sent", accepted: [] };
  const body = completionBody("local-model", fixture, true);
  assert.deepEqual(Object.keys(body).sort(), [
    "chat_template_kwargs",
    "max_tokens",
    "messages",
    "model",
    "reasoning_effort",
    "stream",
    "temperature",
  ]);
  assert.equal(body.messages.length, 2);
  assert.equal(body.messages[1].content.includes(fixture.input), true);

  const serialized = JSON.stringify(body);
  for (const forbidden of [
    "audio",
    "clipboard",
    "window_title",
    "window title",
    "application_data",
    "application data",
    "foreground_app",
  ]) {
    assert.equal(serialized.toLowerCase().includes(forbidden), false, `${forbidden} must not reach cleanup`);
  }
});

test("cleanup output normalization accepts text and rejects wrappers", () => {
  assert.equal(normalizeCleanupOutput("<think>reasoning</think>  Clean text.  "), "Clean text.");
  assert.equal(normalizeCleanupOutput('"Clean text."'), null);
  assert.equal(normalizeCleanupOutput("```text\nClean text.\n```"), null);
  assert.equal(normalizeCleanupOutput('{"transcription":"Clean text."}'), null);
});

test("nearest-rank percentiles stay deterministic for small benchmark samples", () => {
  assert.equal(percentile([40, 10, 30, 20], 50), 20);
  assert.equal(percentile([40, 10, 30, 20], 95), 40);
  assert.equal(percentile([], 50), null);
});

test("short dictation evidence reports repeatable per-fixture p50 and p95", () => {
  const measurements = [];
  for (let run = 1; run <= 3; run += 1) {
    for (const fixture of ["punctuation", "number_words", "spoken_punctuation", "filler_question", "instruction_text"]) {
      measurements.push({
        fixture,
        run,
        latency_ms: fixture === SHORT_DICTATION_FIXTURE_ID ? run * 10 : 5,
        success: true,
        correct: true,
        error: null,
      });
    }
  }
  const summary = summarizeCandidate(
    { label: "fixture-model", modelPath: "/models/fixture.gguf", profile: "generic-v1" },
    measurements,
    {},
  );
  assert.deepEqual(summary.short_dictation_fixture, {
    id: "punctuation",
    cleanup_latency_ms: { p50: 20, p95: 30, max: 30 },
  });
  assert.equal(JSON.stringify(summary.short_dictation_fixture).includes("please send"), false);
});

test("s1-mini profile uses its trained control contract and greedy decoding", async (t) => {
  let body;
  const server = createServer((request, response) => {
    if (request.url !== "/v1/chat/completions") {
      response.statusCode = 404;
      response.end();
      return;
    }
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ choices: [{ message: { content: "Clean text." } }] }));
    });
  });
  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  t.after(() => server.close());
  const address = server.address();
  const fixture = { id: "local", input: "clean text", accepted: ["Clean text."] };
  const result = await runFixture(
    `http://127.0.0.1:${address.port}/v1`,
    "loaded-model",
    fixture,
    1_000,
    "s1-mini-v1",
  );

  assert.equal(result.correct, true);
  assert.equal(body.temperature, 0);
  assert.deepEqual(body.chat_template_kwargs, { enable_thinking: false });
  assert.equal(body.reasoning_effort, undefined);
  assert.match(body.messages[0].content, /^You are a text normalizer/);
  assert.match(body.messages[1].content, /^\[Styling: semi-formal\] \[Structure: prose\] \[Context: general\]\nclean text$/);
});

test("fixture execution measures loopback cleanup without persisting output text", async (t) => {
  const server = createServer((request, response) => {
    if (request.url === "/v1/chat/completions") {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ choices: [{ message: { content: "Clean text." } }] }));
      return;
    }
    response.statusCode = 404;
    response.end();
  });
  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  t.after(() => server.close());
  const address = server.address();
  const fixture = { id: "local", input: "clean text", accepted: ["Clean text."] };
  const result = await runFixture(`http://127.0.0.1:${address.port}/v1`, "loaded-model", fixture, 1_000);

  assert.equal(result.success, true);
  assert.equal(result.correct, true);
  assert.equal(typeof result.latency_ms, "number");
  assert.equal(JSON.stringify(result).includes("Clean text."), false);
  assert.equal(JSON.stringify(result).includes("clean text"), false);
});
