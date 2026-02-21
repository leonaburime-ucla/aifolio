import type { ChartSpec } from "@/features/ai/types/chart.types";

export type CopilotAssistantPayload = {
  message: string;
  chartSpec: ChartSpec | ChartSpec[] | null;
};

const ALLOWED_CHART_TYPES: ReadonlySet<ChartSpec["type"]> = new Set([
  "line",
  "area",
  "bar",
  "scatter",
  "histogram",
  "density",
  "roc",
  "pr",
  "errorbar",
  "heatmap",
  "box",
  "violin",
  "biplot",
  "dendrogram",
  "surface",
]);

const DEBUG_COPILOT_PAYLOAD =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG === "1" ||
  process.env.COPILOT_DEBUG === "1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stripCodeFences(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("```")) return trimmed;
  return trimmed.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
}

function parseJsonObject(raw: string): Record<string, unknown> | null {
  const cleaned = stripCodeFences(raw);
  try {
    const parsed = JSON.parse(cleaned);
    return isRecord(parsed) ? parsed : null;
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start < 0 || end < 0 || end <= start) return null;
    try {
      const parsed = JSON.parse(cleaned.slice(start, end + 1));
      return isRecord(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }
}

function normalizeChartSpec(raw: unknown, fallbackId: string): ChartSpec | null {
  if (!isRecord(raw)) return null;

  const type = raw.type;
  const xKey = raw.xKey;
  const yKeys = raw.yKeys;
  const data = raw.data;

  if (typeof type !== "string" || !ALLOWED_CHART_TYPES.has(type as ChartSpec["type"])) return null;
  if (typeof xKey !== "string" || !xKey.trim()) return null;
  if (!Array.isArray(yKeys)) return null;
  if (!Array.isArray(data)) return null;

  const normalizedYKeys = yKeys.filter((key): key is string => typeof key === "string" && !!key.trim());
  const normalizedData = data.filter(isRecord) as Array<Record<string, string | number>>;

  if (!normalizedYKeys.length || !normalizedData.length) return null;

  const spec: ChartSpec = {
    id: typeof raw.id === "string" && raw.id.trim() ? raw.id : fallbackId,
    title: typeof raw.title === "string" && raw.title.trim() ? raw.title : `${type} chart`,
    type: type as ChartSpec["type"],
    xKey,
    yKeys: normalizedYKeys,
    data: normalizedData,
  };

  if (typeof raw.description === "string" && raw.description.trim()) spec.description = raw.description;
  if (typeof raw.xLabel === "string" && raw.xLabel.trim()) spec.xLabel = raw.xLabel;
  if (typeof raw.yLabel === "string" && raw.yLabel.trim()) spec.yLabel = raw.yLabel;
  if (typeof raw.zKey === "string" && raw.zKey.trim()) spec.zKey = raw.zKey;
  if (typeof raw.colorKey === "string" && raw.colorKey.trim()) spec.colorKey = raw.colorKey;
  if (typeof raw.unit === "string" && raw.unit.trim()) spec.unit = raw.unit;
  if (typeof raw.currency === "string" && raw.currency.trim()) spec.currency = raw.currency;

  if (isRecord(raw.timeframe) && typeof raw.timeframe.start === "string" && typeof raw.timeframe.end === "string") {
    spec.timeframe = { start: raw.timeframe.start, end: raw.timeframe.end };
  }

  if (isRecord(raw.source) && typeof raw.source.provider === "string") {
    spec.source = {
      provider: raw.source.provider,
      ...(typeof raw.source.url === "string" ? { url: raw.source.url } : {}),
    };
  }

  if (isRecord(raw.meta)) {
    const meta: ChartSpec["meta"] = {};
    if (typeof raw.meta.datasetLabel === "string") meta.datasetLabel = raw.meta.datasetLabel;
    if (typeof raw.meta.queryTimeMs === "number") meta.queryTimeMs = raw.meta.queryTimeMs;
    if (Object.keys(meta).length) spec.meta = meta;
  }

  if (isRecord(raw.errorKeys)) {
    const errorKeys = Object.entries(raw.errorKeys).reduce<Record<string, string>>((acc, [k, v]) => {
      if (typeof v === "string" && k.trim() && v.trim()) acc[k] = v;
      return acc;
    }, {});
    if (Object.keys(errorKeys).length) spec.errorKeys = errorKeys;
  }

  return spec;
}

function normalizeChartSpecPayload(raw: unknown): ChartSpec | ChartSpec[] | null {
  if (raw == null) return null;

  if (Array.isArray(raw)) {
    const specs = raw
      .map((entry, index) => normalizeChartSpec(entry, `chart_${index + 1}`))
      .filter((entry): entry is ChartSpec => entry !== null);
    return specs.length ? specs : null;
  }

  return normalizeChartSpec(raw, "chart_1");
}

/**
 * Validate and normalize unknown data into a chart spec payload.
 *
 * Intended for frontend tool handlers that receive model arguments and need
 * to enforce the same schema guarantees as assistant-message parsing.
 */
export function normalizeChartSpecInput(raw: unknown): ChartSpec | ChartSpec[] | null {
  return normalizeChartSpecPayload(raw);
}

export function parseCopilotAssistantPayload(rawContent: string): CopilotAssistantPayload | null {
  const parsed = parseJsonObject(rawContent);
  if (!parsed) return null;

  const message = typeof parsed.message === "string" ? parsed.message : "";
  if (!message.trim()) return null;

  const payload = {
    message,
    chartSpec: normalizeChartSpecPayload(parsed.chartSpec),
  };
  const chartSpec = payload.chartSpec;
  if (DEBUG_COPILOT_PAYLOAD) {
    console.log("[copilot-payload] parse.success", {
      hasChartSpec: chartSpec !== null,
      chartCount: Array.isArray(chartSpec) ? chartSpec.length : chartSpec ? 1 : 0,
      chartTypes: Array.isArray(chartSpec)
        ? chartSpec.map((spec) => spec.type)
        : chartSpec
          ? [chartSpec.type]
          : [],
      messagePreview: payload.message.slice(0, 180),
      chartSpec,
    });
  }
  return payload;
}

export function extractCopilotDisplayMessage(rawContent: string): string {
  const parsed = parseCopilotAssistantPayload(rawContent);
  if (!parsed) return rawContent;
  return parsed.message;
}
