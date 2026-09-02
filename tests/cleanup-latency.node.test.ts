import assert from "node:assert/strict";
import test from "node:test";
import {
  isCleanupCompletionLog,
  isCleanupStartLog,
  summarizeCleanupLatencies,
} from "../src/lib/cleanupLatencyStats.ts";

test("recognizes existing post-processing lifecycle logs", () => {
  assert.equal(
    isCleanupStartLog(
      "Starting LLM post-processing with provider 'custom' (model: qwen)",
    ),
    true,
  );
  assert.equal(
    isCleanupCompletionLog(
      "LLM post-processing succeeded for provider 'custom'. Output length: 42 chars",
    ),
    true,
  );
  assert.equal(
    isCleanupCompletionLog(
      "LLM post-processing failed for provider 'custom': timeout. Falling back to original transcription.",
    ),
    true,
  );
});

test("reports nearest-rank p50 and p95 from valid samples", () => {
  assert.deepEqual(
    summarizeCleanupLatencies([
      100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    ]),
    {
      count: 10,
      latestMs: 1000,
      p50Ms: 500,
      p95Ms: 1000,
    },
  );
});

test("ignores invalid samples and returns null without measurements", () => {
  assert.equal(summarizeCleanupLatencies([Number.NaN, -1]), null);
  assert.deepEqual(
    summarizeCleanupLatencies([300, Number.POSITIVE_INFINITY, 100]),
    {
      count: 2,
      latestMs: 100,
      p50Ms: 100,
      p95Ms: 300,
    },
  );
});
