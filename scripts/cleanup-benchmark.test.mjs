import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import {
  assertLoopbackEndpoint,
  candidateIdentity,
  extractModelPath,
  normalizeCleanupOutput,
  parseArgs,
  parseCandidateSpec,
  percentile,
  replaceModelPath,
  runFixture,
} from "./cleanup-benchmark.mjs";

test("candidate specs separate reporting labels from local asset paths", () => {
  const candidate = parseCandidateSpec("qwen3-1.7b=./models/qwen.gguf");
  assert.equal(candidate.label, "qwen3-1.7b");
  assert.equal(candidate.modelPath.endsWith("models/qwen.gguf"), true);
  assert.deepEqual(candidateIdentity(candidate), {
    candidate: "qwen3-1.7b",
    model_asset: "qwen.gguf",
  });
  assert.equal(JSON.stringify(candidateIdentity(candidate)).includes(candidate.modelPath), false);
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
