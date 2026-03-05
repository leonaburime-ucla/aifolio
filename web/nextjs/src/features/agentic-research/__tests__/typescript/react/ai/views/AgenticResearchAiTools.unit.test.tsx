import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";

const useCopilotActionMock = vi.fn();
const handlers = {
  addChartSpec: vi.fn(() => ({ status: "ok" })),
  clearCharts: vi.fn(() => ({ status: "ok" })),
  removeChartSpec: vi.fn(() => ({ status: "ok" })),
  reorderChartSpecs: vi.fn(() => ({ status: "ok" })),
  setActiveDataset: vi.fn(() => ({ status: "ok" })),
};

vi.mock("@copilotkit/react-core", () => ({
  useCopilotAction: (...args: unknown[]) => useCopilotActionMock(...args),
}));

vi.mock("@/features/agentic-research/typescript/react/ai/adapters/useAgenticResearchAiSurface", () => ({
  useAgenticResearchAiSurface: () => ({ handlers }),
}));

import AgenticResearchAiTools from "@/features/agentic-research/typescript/react/ai/views/AgenticResearchAiTools";

describe("AgenticResearchAiTools", () => {
  it("registers all copilot tools and normalizes handler args", () => {
    render(<AgenticResearchAiTools />);

    expect(useCopilotActionMock).toHaveBeenCalledTimes(8);
    const registrations = useCopilotActionMock.mock.calls.map((call) => call[0]);
    const byName = Object.fromEntries(registrations.map((entry) => [entry.name, entry]));

    byName["ar-add_chart_spec"].handler({ chartSpec: { id: "a" } });
    byName["ar-clear_charts"].handler();
    byName["ar-remove_chart_spec"].handler({ chart_id: "a" });
    byName["remove_chart_spec"].handler({ chartId: "b" });
    byName["remove_chart_spec"].handler({ id: "c" });
    byName["remove_chart_spec"].handler({});
    byName["ar-reorder_chart_specs"].handler({ ordered_ids: ["a"], from_index: 0, to_index: 1 });
    byName["reorder_chart_specs"].handler({ orderedIds: ["b"], fromIndex: 1, toIndex: 0 });
    byName["ar-set_active_dataset"].handler({ dataset_id: "iris" });
    byName["set_active_dataset"].handler({ dataset: "wine" });
    byName["set_active_dataset"].handler({ id: "digits" });
    byName["set_active_dataset"].handler({});

    expect(handlers.addChartSpec).toHaveBeenCalledWith({ chartSpec: { id: "a" }, chartSpecs: undefined });
    expect(handlers.clearCharts).toHaveBeenCalledTimes(1);
    expect(handlers.removeChartSpec).toHaveBeenNthCalledWith(1, "a");
    expect(handlers.removeChartSpec).toHaveBeenNthCalledWith(2, "b");
    expect(handlers.removeChartSpec).toHaveBeenNthCalledWith(3, "c");
    expect(handlers.removeChartSpec).toHaveBeenNthCalledWith(4, "");
    expect(handlers.reorderChartSpecs).toHaveBeenNthCalledWith(1, {
      ordered_ids: ["a"],
      from_index: 0,
      to_index: 1,
    });
    expect(handlers.reorderChartSpecs).toHaveBeenNthCalledWith(2, {
      ordered_ids: ["b"],
      from_index: 1,
      to_index: 0,
    });
    expect(handlers.setActiveDataset).toHaveBeenNthCalledWith(1, "iris");
    expect(handlers.setActiveDataset).toHaveBeenNthCalledWith(2, "wine");
    expect(handlers.setActiveDataset).toHaveBeenNthCalledWith(3, "digits");
    expect(handlers.setActiveDataset).toHaveBeenNthCalledWith(4, "");
  });
});
