import { describe, expect, it } from "vitest";
import {
  findOptimalParamsFromRuns,
  formatBytes,
  formatInt,
  formatMetricNumber,
  parseNumericValue,
} from "@/features/ml/typescript/utils";

describe("utils/index", () => {
  it("re-exports utility functions", () => {
    expect(formatBytes({ value: 1024 })).toContain("KB");
    expect(formatInt({ value: 1234 })).toBe("1,234");
    expect(formatMetricNumber({ value: 0.1234567 })).toBe("0.12346");
    expect(parseNumericValue({ value: "1.5" })).toBe(1.5);
    expect(findOptimalParamsFromRuns({ rows: [] })).toBeNull();
  });
});
