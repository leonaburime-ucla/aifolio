import type {
  CoerceNumberParams,
  FormatValueParams,
  FormatXAxisValueParams,
} from "@/features/recharts/__types__/typescript/logic/chartFormatting.types";

export const SERIES_COLORS = ["#18181b", "#2563eb", "#10b981", "#f59e0b", "#ef4444"] as const;

/**
 * Coerce string-like numeric values into numbers for chart rendering.
 * @param params - Required parameters.
 * @param params.value - Raw datum value.
 * @returns Parsed number when possible, otherwise the original value.
 */
export function coerceNumber({ value }: CoerceNumberParams): number | unknown {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : value;
  }
  return value;
}

/**
 * Format axis and tooltip values using chart-level currency/unit metadata.
 * @param params - Required parameters.
 * @param params.value - Value to format.
 * @param params.spec - Chart specification with formatting metadata.
 * @returns Formatted label string.
 */
export function formatValue({ value, spec }: FormatValueParams): string {
  if (typeof value !== "number") return String(value ?? "");
  if (spec.currency) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: spec.currency,
      maximumFractionDigits: 2,
    }).format(value);
  }
  if (spec.unit) return `${value} ${spec.unit}`;
  if (!Number.isInteger(value)) {
    return parseFloat(value.toFixed(7)).toString();
  }
  return String(value);
}

/**
 * Format x-axis tick values for numeric scatter-style charts.
 * @param params - Required parameters.
 * @param params.value - Tick value.
 * @param params.spec - Chart spec used to infer formatting mode.
 * @returns Formatted x-axis label.
 */
export function formatXAxisValue({ value, spec }: FormatXAxisValueParams): string {
  const isYearLikeAxis =
    typeof spec.xKey === "string" &&
    /(year|yr|date)/i.test(spec.xKey);

  if (typeof value === "number" && value >= 1000 && value <= 3000) {
    return String(Math.round(value));
  }

  if (typeof value === "string" && isYearLikeAxis) {
    const parsedDate = new Date(value);
    if (!Number.isNaN(parsedDate.getTime())) {
      return new Intl.DateTimeFormat("en-US", {
        month: "short",
        year: "numeric",
      }).format(parsedDate);
    }
  }

  if (typeof value === "number" && (spec.type === "scatter" || spec.type === "biplot")) {
    const fixed = value.toFixed(3);
    return fixed === "-0.000" ? "0.000" : fixed;
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return new Intl.NumberFormat("en-US", {
        maximumFractionDigits: 0,
      }).format(value);
    }
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 2,
    }).format(value);
  }
  return String(value ?? "");
}
