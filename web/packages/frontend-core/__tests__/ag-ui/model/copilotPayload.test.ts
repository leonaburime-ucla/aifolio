import { describe, expect, it } from "vitest";
import {
  isRecord,
  stripCodeFences,
  parseJsonObject,
  normalizeChartSpec,
  normalizeChartSpecInput,
  parseCopilotAssistantPayload,
  extractCopilotDisplayMessage,
} from "../../../src/ag-ui/model/copilotPayload";

describe("isRecord", () => {
  it("returns true for plain objects", () => {
    expect(isRecord({})).toBe(true);
    expect(isRecord({ key: "value" })).toBe(true);
  });

  it("returns false for non-objects", () => {
    expect(isRecord(null)).toBe(false);
    expect(isRecord(undefined)).toBe(false);
    expect(isRecord(42)).toBe(false);
    expect(isRecord("string")).toBe(false);
    expect(isRecord([1, 2])).toBe(false);
  });
});

describe("stripCodeFences", () => {
  it("returns plain content unchanged", () => {
    expect(stripCodeFences('{"a":1}')).toBe('{"a":1}');
  });

  it("strips ```json fences", () => {
    expect(stripCodeFences('```json\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it("strips ``` fences without language", () => {
    expect(stripCodeFences('```\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it("handles whitespace around fences", () => {
    expect(stripCodeFences('  ```json\n  {"a":1}  \n```  ')).toBe('{"a":1}');
  });
});

describe("parseJsonObject", () => {
  it("parses valid JSON object", () => {
    expect(parseJsonObject('{"message":"hi"}')).toEqual({ message: "hi" });
  });

  it("returns null for non-object JSON", () => {
    expect(parseJsonObject("[1,2,3]")).toBe(null);
    expect(parseJsonObject('"hello"')).toBe(null);
  });

  it("extracts object from surrounding text", () => {
    expect(parseJsonObject('some text {"a":1} trailing')).toEqual({ a: 1 });
  });

  it("returns null for completely unparseable text", () => {
    expect(parseJsonObject("not json at all")).toBe(null);
  });

  it("returns null when extracted JSON-like content is malformed", () => {
    expect(parseJsonObject("prefix {bad json} suffix")).toBe(null);
  });

  it("strips code fences before parsing", () => {
    expect(parseJsonObject('```json\n{"key":"val"}\n```')).toEqual({ key: "val" });
  });
});

describe("normalizeChartSpec", () => {
  const validSpec = {
    type: "bar",
    xKey: "x",
    yKeys: ["y"],
    data: [{ x: 1, y: 2 }],
  };

  it("returns a ChartSpec for valid input", () => {
    const result = normalizeChartSpec(validSpec, "fallback-id");
    expect(result).not.toBeNull();
    expect(result!.id).toBe("fallback-id");
    expect(result!.type).toBe("bar");
    expect(result!.title).toBe("bar chart");
  });

  it("uses provided id and title when present", () => {
    const result = normalizeChartSpec({ ...validSpec, id: "my-id", title: "My Chart" }, "fallback");
    expect(result!.id).toBe("my-id");
    expect(result!.title).toBe("My Chart");
  });

  it("returns null for non-record input", () => {
    expect(normalizeChartSpec(null, "id")).toBe(null);
    expect(normalizeChartSpec("string", "id")).toBe(null);
  });

  it("returns null for invalid type", () => {
    expect(normalizeChartSpec({ ...validSpec, type: "invalid" }, "id")).toBe(null);
  });

  it("returns null for missing xKey", () => {
    expect(normalizeChartSpec({ ...validSpec, xKey: "" }, "id")).toBe(null);
  });

  it("returns null for non-array yKeys", () => {
    expect(normalizeChartSpec({ ...validSpec, yKeys: "y" }, "id")).toBe(null);
  });

  it("returns null for empty data array", () => {
    expect(normalizeChartSpec({ ...validSpec, data: [] }, "id")).toBe(null);
  });

  it("includes optional fields when present", () => {
    const result = normalizeChartSpec(
      { ...validSpec, description: "desc", xLabel: "X", yLabel: "Y", zKey: "z", colorKey: "c", unit: "kg", currency: "USD" },
      "id"
    );
    expect(result!.description).toBe("desc");
    expect(result!.xLabel).toBe("X");
    expect(result!.yLabel).toBe("Y");
    expect(result!.zKey).toBe("z");
    expect(result!.colorKey).toBe("c");
    expect(result!.unit).toBe("kg");
    expect(result!.currency).toBe("USD");
  });

  it("includes timeframe when valid", () => {
    const result = normalizeChartSpec(
      { ...validSpec, timeframe: { start: "2024-01", end: "2024-12" } },
      "id"
    );
    expect(result!.timeframe).toEqual({ start: "2024-01", end: "2024-12" });
  });

  it("includes source when valid", () => {
    const result = normalizeChartSpec(
      { ...validSpec, source: { provider: "api", url: "http://x" } },
      "id"
    );
    expect(result!.source).toEqual({ provider: "api", url: "http://x" });
  });

  it("includes meta when valid", () => {
    const result = normalizeChartSpec(
      { ...validSpec, meta: { datasetLabel: "test", queryTimeMs: 42 } },
      "id"
    );
    expect(result!.meta).toEqual({ datasetLabel: "test", queryTimeMs: 42 });
  });

  it("includes errorKeys when valid", () => {
    const result = normalizeChartSpec(
      { ...validSpec, errorKeys: { y: "y_err" } },
      "id"
    );
    expect(result!.errorKeys).toEqual({ y: "y_err" });
  });
});

describe("normalizeChartSpecInput", () => {
  const validSpec = {
    type: "line",
    xKey: "x",
    yKeys: ["y"],
    data: [{ x: 1, y: 2 }],
  };

  it("normalizes a single spec", () => {
    const result = normalizeChartSpecInput(validSpec);
    expect(result).not.toBeNull();
    expect((result as { id: string }).id).toBe("chart_1");
  });

  it("normalizes an array of specs", () => {
    const result = normalizeChartSpecInput([validSpec, validSpec]);
    expect(Array.isArray(result)).toBe(true);
    expect((result as unknown[]).length).toBe(2);
  });

  it("returns null for null input", () => {
    expect(normalizeChartSpecInput(null)).toBe(null);
  });

  it("returns null for array of invalid specs", () => {
    expect(normalizeChartSpecInput([{ invalid: true }])).toBe(null);
  });
});

describe("parseCopilotAssistantPayload", () => {
  it("parses valid payload with message", () => {
    const result = parseCopilotAssistantPayload(JSON.stringify({ message: "Hello" }));
    expect(result).not.toBeNull();
    expect(result!.message).toBe("Hello");
  });

  it("returns null for empty message", () => {
    expect(parseCopilotAssistantPayload(JSON.stringify({ message: "  " }))).toBe(null);
  });

  it("returns null for unparseable content", () => {
    expect(parseCopilotAssistantPayload("not json")).toBe(null);
  });

  it("includes chartSpec when present", () => {
    const raw = JSON.stringify({
      message: "Here's a chart",
      chartSpec: { type: "bar", xKey: "x", yKeys: ["y"], data: [{ x: 1, y: 2 }] },
    });
    const result = parseCopilotAssistantPayload(raw);
    expect(result!.chartSpec).not.toBeNull();
  });

  it("sets type to TextMessage when specified", () => {
    const raw = JSON.stringify({ message: "hi", type: "TextMessage" });
    const result = parseCopilotAssistantPayload(raw);
    expect(result!.type).toBe("TextMessage");
  });
});

describe("extractCopilotDisplayMessage", () => {
  it("extracts message from valid payload", () => {
    expect(extractCopilotDisplayMessage(JSON.stringify({ message: "Hello world" })))
      .toBe("Hello world");
  });

  it("returns raw content when parsing fails", () => {
    expect(extractCopilotDisplayMessage("plain text")).toBe("plain text");
  });
});
