import { describe, expect, it } from "vitest";
import {
  SERIES_COLORS,
  coerceNumber,
  formatValue,
  formatXAxisValue,
} from "@/features/recharts/typescript/logic/chartFormatting.logic";

const baseSpec = {
  id: "c1",
  title: "Chart",
  type: "line" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
};

describe("chartFormatting.logic", () => {
  it("coerces numeric strings and preserves non-numeric values", () => {
    expect(coerceNumber({ value: 10 })).toBe(10);
    expect(coerceNumber({ value: " 12.5 " })).toBe(12.5);
    expect(coerceNumber({ value: "" })).toBe("");
    expect(coerceNumber({ value: "abc" })).toBe("abc");
    expect(coerceNumber({ value: null })).toBeNull();
  });

  it("formats values with currency, unit, float and fallback modes", () => {
    expect(formatValue({ value: 12.345, spec: { ...baseSpec, currency: "USD" } })).toBe(
      "$12.35"
    );
    expect(formatValue({ value: 5, spec: { ...baseSpec, unit: "kg" } })).toBe("5 kg");
    expect(formatValue({ value: 0.123456789, spec: baseSpec })).toBe("0.1234568");
    expect(formatValue({ value: 3, spec: baseSpec })).toBe("3");
    expect(formatValue({ value: "raw", spec: baseSpec })).toBe("raw");
    expect(formatValue({ value: null, spec: baseSpec })).toBe("");
  });

  it("formats x-axis values for year-like, scatter and numeric axes", () => {
    expect(formatXAxisValue({ value: 2024.4, spec: baseSpec })).toBe("2024");
    expect(
      formatXAxisValue({
        value: "2024-01-15",
        spec: { ...baseSpec, xKey: "year_date" },
      })
    ).toContain("2024");
    expect(
      formatXAxisValue({
        value: 1.23456,
        spec: { ...baseSpec, type: "scatter" },
      })
    ).toBe("1.235");
    expect(
      formatXAxisValue({
        value: -0.0004,
        spec: { ...baseSpec, type: "biplot" },
      })
    ).toBe("0.000");
    expect(formatXAxisValue({ value: 10000, spec: baseSpec })).toBe("10,000");
    expect(formatXAxisValue({ value: 3.456, spec: baseSpec })).toBe("3.46");
    expect(formatXAxisValue({ value: "label", spec: baseSpec })).toBe("label");
    expect(formatXAxisValue({ value: null, spec: baseSpec })).toBe("");
  });

  it("exports stable series colors", () => {
    expect(SERIES_COLORS.length).toBeGreaterThan(3);
    expect(SERIES_COLORS[0]).toBe("#18181b");
  });
});
