import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ChartSpec } from "@/features/ai/types/chart.types";
import LandingCharts from "@/core/views/screens/LandingCharts";

vi.mock("@/features/recharts/views/components/ChartRenderer", () => ({
  default: ({ spec }: { spec: ChartSpec }) => (
    <div data-testid="chart-renderer">{spec.id}</div>
  ),
}));

describe("LandingCharts", () => {
  it("renders empty-state message when no charts are present", () => {
    const orchestrator = () => ({
      chartSpecs: [],
      removeChartSpec: vi.fn(),
    });

    render(<LandingCharts orchestrator={orchestrator} />);

    expect(screen.getByText("Charts generated from chat will appear here.")).toBeInTheDocument();
  });

  it("uses injected orchestrator and delegates chart removal", () => {
    const removeChartSpec = vi.fn();
    const orchestrator = vi.fn(() => ({
      chartSpecs: [
        {
          id: "chart-1",
          title: "Chart 1",
          type: "line",
          xKey: "x",
          yKeys: ["y"],
          data: [],
        } satisfies ChartSpec,
      ],
      removeChartSpec,
    }));

    render(<LandingCharts orchestrator={orchestrator} />);

    expect(orchestrator).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Remove chart" }));
    expect(removeChartSpec).toHaveBeenCalledWith("chart-1");
  });
});
