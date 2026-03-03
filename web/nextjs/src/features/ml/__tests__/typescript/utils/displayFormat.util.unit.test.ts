import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatInt,
} from "@/features/ml/typescript/utils/displayFormat.util";

describe("displayFormat.util", () => {
  it("formats bytes across units", () => {
    expect(formatBytes({ value: 10 })).toBe("10 B");
    expect(formatBytes({ value: 2048 })).toBe("2.0 KB");
    expect(formatBytes({ value: 5 * 1024 ** 2 })).toBe("5.00 MB");
    expect(formatBytes({ value: 3 * 1024 ** 3 })).toBe("3.00 GB");
  });

  it("returns n/a for invalid byte values", () => {
    expect(formatBytes({ value: null })).toBe("n/a");
    expect(formatBytes({ value: Number.NaN })).toBe("n/a");
    expect(formatBytes({ value: -1 })).toBe("n/a");
  });

  it("formats integers or returns n/a for invalid values", () => {
    expect(formatInt({ value: 1234567 })).toBe("1,234,567");
    expect(formatInt({ value: undefined })).toBe("n/a");
    expect(formatInt({ value: Number.NaN })).toBe("n/a");
  });
});
