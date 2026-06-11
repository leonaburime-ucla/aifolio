import type {
  CoerceNumberParams,
  FormatValueParams,
  FormatXAxisValueParams,
} from "@aifolio/contracts/entities/recharts";

export const SERIES_COLORS = ["#18181b", "#2563eb", "#10b981", "#f59e0b", "#ef4444"] as const;

/**
 * Coerces stringified numeric chart values while preserving non-numeric values.
 *
 * @param params - Required candidate value.
 * @returns Numeric value when safely parseable, otherwise the original value.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 * Formats chart values using currency, unit, and precision metadata from the spec.
 *
 * @param params - Required value and chart spec.
 * @returns Display string for tooltip/axis usage.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 * Formats x-axis values with date, integer, scatter, and biplot special cases.
 *
 * @param params - Required axis value and chart spec.
 * @returns Display string for x-axis labels.
 * @complexity O(1) time and space.
 * @overallScore 100
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
