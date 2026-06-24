import { describe, it, expect, beforeEach } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";
import {
  parseCopilotAssistantPayload,
  normalizeChartSpecInput,
  extractCopilotDisplayMessage,
} from "@aifolio/frontend-core/ag-ui";
import { useChartStore } from "~/composables/useChartStore";

const chartSpec: ChartSpec = {
  id: "chart-a",
  title: "Revenue",
  type: "line",
  xKey: "month",
  yKeys: ["revenue"],
  data: [{ month: "Jan", revenue: 10 }],
};

function assistantContent(spec: ChartSpec | ChartSpec[] | null = chartSpec): string {
  return JSON.stringify({
    message: "Here is a chart.",
    chartSpec: spec,
  });
}

describe("parseCopilotAssistantPayload", () => {
  it("returns null for non-JSON content", () => {
    expect(parseCopilotAssistantPayload("hello world")).toBeNull();
    expect(parseCopilotAssistantPayload("")).toBeNull();
    expect(parseCopilotAssistantPayload("{")).toBeNull();
  });

  it("returns null when message field is empty or missing", () => {
    expect(parseCopilotAssistantPayload(JSON.stringify({ chartSpec }))).toBeNull();
    expect(parseCopilotAssistantPayload(JSON.stringify({ message: "" }))).toBeNull();
    expect(parseCopilotAssistantPayload(JSON.stringify({ message: "   " }))).toBeNull();
  });

  it("parses valid payload with chart spec", () => {
    const result = parseCopilotAssistantPayload(assistantContent());
    expect(result).not.toBeNull();
    expect(result!.message).toBe("Here is a chart.");
    expect(result!.chartSpec).toEqual(chartSpec);
  });

  it("parses payload without chart spec", () => {
    const result = parseCopilotAssistantPayload(JSON.stringify({ message: "plain answer" }));
    expect(result).not.toBeNull();
    expect(result!.message).toBe("plain answer");
    expect(result!.chartSpec).toBeNull();
  });

  it("strips code fences before parsing", () => {
    const fenced = "```json\n" + assistantContent() + "\n```";
    const result = parseCopilotAssistantPayload(fenced);
    expect(result).not.toBeNull();
    expect(result!.chartSpec).toEqual(chartSpec);
  });

  it("extracts JSON even with surrounding text", () => {
    const wrapped = "Here is the result:\n" + assistantContent() + "\nDone.";
    const result = parseCopilotAssistantPayload(wrapped);
    expect(result).not.toBeNull();
    expect(result!.message).toBe("Here is a chart.");
  });
});

describe("normalizeChartSpecInput", () => {
  it("returns null for null/undefined input", () => {
    expect(normalizeChartSpecInput(null)).toBeNull();
    expect(normalizeChartSpecInput(undefined)).toBeNull();
  });

  it("normalizes a single valid chart spec", () => {
    const result = normalizeChartSpecInput(chartSpec);
    expect(result).toEqual(chartSpec);
  });

  it("normalizes an array of chart specs", () => {
    const specs = [chartSpec, { ...chartSpec, id: "chart-b", type: "bar" as const }];
    const result = normalizeChartSpecInput(specs);
    expect(Array.isArray(result)).toBe(true);
    expect((result as ChartSpec[]).length).toBe(2);
    expect((result as ChartSpec[])[0].id).toBe("chart-a");
    expect((result as ChartSpec[])[1].id).toBe("chart-b");
  });

  it("rejects specs with invalid chart type", () => {
    const invalid = { ...chartSpec, type: "invalid_type" };
    expect(normalizeChartSpecInput(invalid)).toBeNull();
  });

  it("rejects specs with missing required fields", () => {
    expect(normalizeChartSpecInput({ type: "line" })).toBeNull();
    expect(normalizeChartSpecInput({ ...chartSpec, xKey: "" })).toBeNull();
    expect(normalizeChartSpecInput({ ...chartSpec, yKeys: [] })).toBeNull();
    expect(normalizeChartSpecInput({ ...chartSpec, data: [] })).toBeNull();
  });

  it("filters invalid entries from arrays", () => {
    const specs = [chartSpec, { type: "invalid" }, { ...chartSpec, id: "chart-c" }];
    const result = normalizeChartSpecInput(specs);
    expect(Array.isArray(result)).toBe(true);
    expect((result as ChartSpec[]).length).toBe(2);
  });

  it("returns null for arrays where all entries are invalid", () => {
    const specs = [{ type: "invalid" }, { type: "also_invalid" }];
    expect(normalizeChartSpecInput(specs)).toBeNull();
  });
});

describe("extractCopilotDisplayMessage", () => {
  it("returns clean message from valid JSON payload", () => {
    expect(extractCopilotDisplayMessage(assistantContent())).toBe("Here is a chart.");
  });

  it("returns raw content for non-JSON strings", () => {
    expect(extractCopilotDisplayMessage("hello world")).toBe("hello world");
  });

  it("returns raw content for JSON without message field", () => {
    const noMessage = JSON.stringify({ data: 123 });
    expect(extractCopilotDisplayMessage(noMessage)).toBe(noMessage);
  });
});

describe("chart bridge integration (Vue)", () => {
  let chartStore: ReturnType<typeof useChartStore>;

  beforeEach(() => {
    setActivePinia(createPinia());
    chartStore = useChartStore();
  });

  it("routes a single chart spec from assistant message to the store", () => {
    const content = assistantContent();
    const payload = parseCopilotAssistantPayload(content);
    expect(payload?.chartSpec).not.toBeNull();

    const specs = normalizeChartSpecInput(payload!.chartSpec);
    const specArray = Array.isArray(specs) ? specs : specs ? [specs] : [];
    for (const spec of specArray) {
      chartStore.addChartSpec(spec);
    }

    expect(chartStore.chartSpecs.length).toBe(1);
    expect(chartStore.chartSpecs[0].id).toBe("chart-a");
  });

  it("routes multiple chart specs from assistant message to the store", () => {
    const content = assistantContent([
      chartSpec,
      { ...chartSpec, id: "chart-b", type: "bar" as const, title: "Costs" },
    ]);
    const payload = parseCopilotAssistantPayload(content);
    expect(payload?.chartSpec).not.toBeNull();

    const specs = normalizeChartSpecInput(payload!.chartSpec);
    const specArray = Array.isArray(specs) ? specs : specs ? [specs] : [];
    for (const spec of specArray) {
      chartStore.addChartSpec(spec);
    }

    expect(chartStore.chartSpecs.length).toBe(2);
  });

  it("skips messages without chart specs (no store mutation)", () => {
    const content = JSON.stringify({ message: "No charts here." });
    const payload = parseCopilotAssistantPayload(content);
    expect(payload?.chartSpec).toBeNull();

    if (payload?.chartSpec) {
      const specs = normalizeChartSpecInput(payload.chartSpec);
      const specArray = Array.isArray(specs) ? specs : specs ? [specs] : [];
      for (const spec of specArray) {
        chartStore.addChartSpec(spec);
      }
    }

    expect(chartStore.chartSpecs.length).toBe(0);
  });

  it("deduplicates processed messages (idempotent)", () => {
    const processedMessageIds = new Set<string>();
    const msgId = "msg-1";
    const content = assistantContent();

    for (let i = 0; i < 3; i++) {
      if (processedMessageIds.has(msgId)) continue;
      const payload = parseCopilotAssistantPayload(content);
      if (!payload?.chartSpec) continue;
      processedMessageIds.add(msgId);
      const specs = normalizeChartSpecInput(payload.chartSpec);
      const specArray = Array.isArray(specs) ? specs : specs ? [specs] : [];
      for (const spec of specArray) {
        chartStore.addChartSpec(spec);
      }
    }

    expect(chartStore.chartSpecs.length).toBe(1);
  });
});
