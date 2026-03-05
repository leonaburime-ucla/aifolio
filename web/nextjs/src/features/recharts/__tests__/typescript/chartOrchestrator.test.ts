import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useChartOrchestrator } from "@/features/recharts/typescript/react/orchestrators/chartOrchestrator";

describe("useChartOrchestrator", () => {
  it("uses injected chart management port when provided", () => {
    const orchestrator = vi.fn(() => ({
      chartSpecs: [
        { id: "chart-1", title: "Chart 1", type: "line", data: [], xKey: "x", yKeys: ["y"] },
      ],
      removeChartSpec: vi.fn(),
      clearChartSpecs: vi.fn(),
    }));

    const { result } = renderHook(() =>
      useChartOrchestrator({ orchestrator })
    );

    expect(orchestrator).toHaveBeenCalledTimes(1);
    expect(result.current.chartSpecs).toHaveLength(1);
    expect(typeof result.current.removeChartSpec).toBe("function");
  });
});
