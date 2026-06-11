import { describe, it, expect } from "vitest";
import {
  normalizeChatApiResult,
  normalizeTextResult,
  parseJsonPayload,
  createModelFetchErrorResult,
} from "../../../src/chat/index";

describe("parseJsonPayload", () => {
  it("parses valid JSON with message field", () => {
    const json = JSON.stringify({ message: "hello", chartSpec: null });
    const result = parseJsonPayload(json);
    expect(result).toEqual({ message: "hello", chartSpec: null });
  });

  it("returns null for non-JSON string", () => {
    expect(parseJsonPayload("not json")).toBeNull();
  });

  it("returns null for string not starting with {", () => {
    expect(parseJsonPayload("[1,2,3]")).toBeNull();
  });

  it("returns null for invalid JSON", () => {
    expect(parseJsonPayload("{broken")).toBeNull();
  });

  it("returns null when parsed object lacks message field", () => {
    expect(parseJsonPayload(JSON.stringify({ data: 1 }))).toBeNull();
  });

  it("handles whitespace around JSON", () => {
    const json = `  ${JSON.stringify({ message: "hi" })}  `;
    const result = parseJsonPayload(json);
    expect(result?.message).toBe("hi");
  });
});

describe("normalizeTextResult", () => {
  it("returns plain text as message with null chartSpec", () => {
    const result = normalizeTextResult("hello world");
    expect(result).toEqual({ message: "hello world", chartSpec: null });
  });

  it("parses embedded JSON payload from text", () => {
    const json = JSON.stringify({ message: "parsed", chartSpec: null });
    const result = normalizeTextResult(json);
    expect(result.message).toBe("parsed");
  });
});

describe("normalizeChatApiResult", () => {
  it("returns null for undefined result", () => {
    expect(normalizeChatApiResult(undefined)).toBeNull();
  });

  it("normalizes string result", () => {
    const result = normalizeChatApiResult("hello");
    expect(result).toEqual({ message: "hello", chartSpec: null });
  });

  it("normalizes object result with message", () => {
    const result = normalizeChatApiResult({ message: "hi" });
    expect(result?.message).toBe("hi");
    expect(result?.chartSpec).toBeNull();
  });

  it("normalizes object result with chartSpec", () => {
    const spec = { id: "c1", title: "T", type: "bar", xKey: "x", yKeys: ["y"], data: [] };
    const result = normalizeChatApiResult({ message: "done", chartSpec: spec });
    expect(result?.chartSpec).toEqual(spec);
  });

  it("returns null for empty object result", () => {
    expect(normalizeChatApiResult({})).toBeNull();
  });

  it("normalizes array of content parts", () => {
    const result = normalizeChatApiResult([
      { type: "text", text: "part1" },
      { type: "text", text: "part2" },
    ]);
    expect(result?.message).toBe("part1\npart2");
  });

  it("returns null for array with no text", () => {
    expect(normalizeChatApiResult([{ type: "image" }])).toBeNull();
  });

  it("parses JSON embedded in object message field", () => {
    const embedded = JSON.stringify({ message: "inner", chartSpec: null });
    const result = normalizeChatApiResult({ message: embedded });
    expect(result?.message).toBe("inner");
  });
});

describe("createModelFetchErrorResult", () => {
  it("creates a timeout error result", () => {
    const result = createModelFetchErrorResult({
      code: "MODEL_FETCH_TIMEOUT",
      retryable: true,
      message: "Timed out",
    });
    expect(result.status).toBe("error");
    expect(result.error.code).toBe("MODEL_FETCH_TIMEOUT");
    expect(result.error.retryable).toBe(true);
  });

  it("creates a failed error result", () => {
    const result = createModelFetchErrorResult({
      code: "MODEL_FETCH_FAILED",
      retryable: false,
      message: "Network error",
    });
    expect(result.error.code).toBe("MODEL_FETCH_FAILED");
    expect(result.error.retryable).toBe(false);
  });
});
