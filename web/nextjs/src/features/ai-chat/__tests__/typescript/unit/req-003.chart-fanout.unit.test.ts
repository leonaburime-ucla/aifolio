import { describe, expect, it, vi } from "vitest";
import { createOnMessageReceived } from "@/features/ai-chat/typescript/logic/chatComposition.logic";
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

describe("REQ-003 chartSpec fan-out", () => {
  it("calls addChartSpec once per chart in deterministic order", () => {
    const addChartSpec = vi.fn();
    const onMessageReceived = createOnMessageReceived({ addChartSpec });

    const chartA = { chartType: "line", title: "A", data: [] } as ChartSpec;
    const chartB = { chartType: "bar", title: "B", data: [] } as ChartSpec;
    const chartC = { chartType: "pie", title: "C", data: [] } as ChartSpec;

    onMessageReceived({
      message: "charts",
      chartSpec: [chartA, chartB, chartC],
    });

    expect(addChartSpec).toHaveBeenCalledTimes(3);
    expect(addChartSpec.mock.calls).toEqual([[chartA], [chartB], [chartC]]);
  });

  it("handles null and single chart payloads", () => {
    const addChartSpec = vi.fn();
    const onMessageReceived = createOnMessageReceived({ addChartSpec });

    onMessageReceived({ message: "no chart", chartSpec: null });
    expect(addChartSpec).not.toHaveBeenCalled();

    const single = { chartType: "line", title: "One", data: [] } as ChartSpec;
    onMessageReceived({ message: "single", chartSpec: single });
    expect(addChartSpec).toHaveBeenCalledTimes(1);
    expect(addChartSpec).toHaveBeenCalledWith(single);
  });
});
