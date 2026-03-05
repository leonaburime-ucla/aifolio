import { describe, expect, it } from "vitest";
import {
  createModelFetchErrorResult,
  normalizeChatApiResult,
  parseJsonPayload,
} from "@/features/ai-chat/typescript/logic/chatApiNormalization.logic";

describe("chatApiNormalization.logic", () => {
  it("parses valid JSON payload strings", () => {
    expect(parseJsonPayload('{"message":"ok","chartSpec":null}')).toEqual({
      message: "ok",
      chartSpec: null,
    });
  });

  it("returns null for malformed json and json without message", () => {
    expect(parseJsonPayload("{bad}")).toBeNull();
    expect(parseJsonPayload('{"chartSpec":null}')).toBeNull();
  });

  it("normalizes raw string and content-part array payloads", () => {
    expect(normalizeChatApiResult("hello")).toEqual({
      message: "hello",
      chartSpec: null,
    });
    expect(
      normalizeChatApiResult([
        { type: "text", text: "line1" },
        { type: "text", text: "line2" },
      ])
    ).toEqual({
      message: "line1\nline2",
      chartSpec: null,
    });
  });

  it("normalizes object payloads and returns null for empty payload", () => {
    expect(
      normalizeChatApiResult({ message: "hi", chartSpec: null })
    ).toEqual({
      message: "hi",
      chartSpec: null,
    });
    expect(
      normalizeChatApiResult({
        message: '{"message":"inner"}',
        chartSpec: [{ chartType: "bar", x: "x", y: "y", data: [] }],
      })
    ).toEqual({
      message: "inner",
      chartSpec: null,
    });
    expect(normalizeChatApiResult({})).toBeNull();
    expect(normalizeChatApiResult(null as unknown as never)).toBeNull();
    expect(normalizeChatApiResult(123 as unknown as never)).toBeNull();
  });

  it("creates deterministic model fetch error objects", () => {
    expect(
      createModelFetchErrorResult({
        code: "MODEL_FETCH_TIMEOUT",
        retryable: true,
        message: "timeout",
      })
    ).toEqual({
      status: "error",
      error: {
        code: "MODEL_FETCH_TIMEOUT",
        retryable: true,
        message: "timeout",
      },
    });
  });
});
