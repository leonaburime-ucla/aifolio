import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const removeChartSpec = vi.fn();
const clearChartSpecs = vi.fn();

vi.mock("@/features/recharts/typescript/react/state/zustand/chartStore", () => ({
  useChartStore: (
    selector: (
      value: {
        chartSpecs: { id: string }[];
        removeChartSpec: typeof removeChartSpec;
        clearChartSpecs: typeof clearChartSpecs;
      }
    ) => unknown
  ) =>
    selector({
      chartSpecs: [{ id: "a" }],
      removeChartSpec,
      clearChartSpecs,
    }),
}));

import { useChartManagementAdapter } from "@/features/recharts/typescript/react/state/adapters/chartManagement.adapter";

describe("chartManagement.adapter", () => {
  it("exposes chart specs plus remove/clear actions", () => {
    const { result } = renderHook(() => useChartManagementAdapter());

    expect(result.current.chartSpecs).toEqual([{ id: "a" }]);
    result.current.removeChartSpec("a");
    result.current.clearChartSpecs();
    expect(removeChartSpec).toHaveBeenCalledWith("a");
    expect(clearChartSpecs).toHaveBeenCalledTimes(1);
  });
});
