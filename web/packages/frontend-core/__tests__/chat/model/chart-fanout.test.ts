import { describe, expect, it, vi } from "vitest";
import { createOnMessageReceived } from "@aifolio/frontend-core/chat";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";

describe("REQ-003 chartSpec fan-out", () => {
  it("calls addChartSpec once per chart in deterministic order", () => {
    const addChartSpec = vi.fn();
    const onMessageReceived = createOnMessageReceived({ addChartSpec });

    const chartA: ChartSpec = { id: "a", type: "line", title: "A", xKey: "x", yKeys: ["y"], data: [] };
    const chartB: ChartSpec = { id: "b", type: "bar", title: "B", xKey: "x", yKeys: ["y"], data: [] };
    const chartC: ChartSpec = { id: "c", type: "scatter", title: "C", xKey: "x", yKeys: ["y"], data: [] };

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

    const single: ChartSpec = { id: "s", type: "line", title: "One", xKey: "x", yKeys: ["y"], data: [] };
    onMessageReceived({ message: "single", chartSpec: single });
    expect(addChartSpec).toHaveBeenCalledTimes(1);
    expect(addChartSpec).toHaveBeenCalledWith(single);
  });
});
