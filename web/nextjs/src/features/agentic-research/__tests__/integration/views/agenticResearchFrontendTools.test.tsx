import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import AgenticResearchFrontendTools from "@/features/agentic-research/views/components/AgenticResearchFrontendTools";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";
import { agenticResearchStore } from "@/features/agentic-research/state/zustand/agenticResearchStore";

type ToolConfig = {
  name: string;
  handler: (args: any) => any;
};

const registry: ToolConfig[] = [];

vi.mock("@copilotkit/react-core", () => ({
  useCopilotAction: (config: ToolConfig) => {
    registry.push(config);
  },
}));

describe("AgenticResearchFrontendTools", () => {
  beforeEach(() => {
    registry.length = 0;
    useAgenticResearchChartStore.setState({ chartSpecs: [] });
    agenticResearchStore.setState({
      datasetManifest: [
        { id: "dataset-a", label: "Dataset A" },
        { id: "dataset-b", label: "Dataset B" },
      ],
      selectedDatasetId: "dataset-a",
      sklearnTools: [],
      tableRows: [],
      tableColumns: [],
      numericMatrix: [],
      featureNames: [],
      pcaChartSpec: null,
      isLoading: false,
      error: null,
    });
  });

  it("returns CHART_NOT_FOUND for missing chart remove request", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "c1", type: "line", data: [], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const removeTool = registry.find((tool) => tool.name === "ar-remove_chart_spec");

    expect(removeTool).toBeDefined();
    const result = removeTool!.handler({ chart_id: "missing" });

    expect(result.status).toBe("error");
    expect(result.code).toBe("CHART_NOT_FOUND");
    expect(result.available_chart_ids).toEqual(["c1"]);
  });

  it("returns INVALID_REORDER_PAYLOAD when reorder payload is invalid", () => {
    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();
    const result = reorderTool!.handler({});

    expect(result.status).toBe("error");
    expect(result.code).toBe("INVALID_REORDER_PAYLOAD");
  });

  it("clears charts and sets selected dataset on valid dataset switch", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "c1", type: "line", data: [], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const setDatasetTool = registry.find((tool) => tool.name === "ar-set_active_dataset");

    expect(setDatasetTool).toBeDefined();
    const result = setDatasetTool!.handler({ dataset_id: "dataset-b" });

    expect(result.status).toBe("ok");
    expect(result.active_dataset_id).toBe("dataset-b");
    expect(useAgenticResearchChartStore.getState().chartSpecs).toEqual([]);
    expect(agenticResearchStore.getState().selectedDatasetId).toBe("dataset-b");
  });

  // NEW: Test for clear_charts alias handler (lines 92-103)
  it("clears all charts via clear_charts alias tool", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 2 }], xKey: "x", yKeys: ["y"] },
        { id: "c2", title: "Chart 2", type: "bar", data: [{ x: 1, y: 3 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const clearChartsTool = registry.find((tool) => tool.name === "clear_charts");

    expect(clearChartsTool).toBeDefined();
    const result = clearChartsTool!.handler({});

    expect(result.status).toBe("ok");
    expect(result.cleared).toBe(true);
    expect(useAgenticResearchChartStore.getState().chartSpecs).toEqual([]);
  });

  // NEW: Test for add_chart_spec alias handler (lines 53-77)
  it("adds a chart spec via add_chart_spec alias tool", () => {
    render(<AgenticResearchFrontendTools />);
    const addChartSpecTool = registry.find((tool) => tool.name === "add_chart_spec");

    expect(addChartSpecTool).toBeDefined();

    const chartSpec = {
      id: "test-chart",
      title: "Test Chart",
      type: "line",
      xKey: "x",
      yKeys: ["y"],
      data: [{ x: 1, y: 10 }, { x: 2, y: 20 }],
    };

    const result = addChartSpecTool!.handler({ chartSpec });

    expect(result.status).toBe("ok");
    expect(result.addedCount).toBe(1);
    expect(result.ids).toContain("test-chart");
    expect(useAgenticResearchChartStore.getState().chartSpecs).toHaveLength(1);
    expect(useAgenticResearchChartStore.getState().chartSpecs[0].id).toBe("test-chart");
  });

  // NEW: Test for add_chart_spec alias with multiple charts via chartSpecs array
  it("adds multiple chart specs via add_chart_spec alias tool using chartSpecs array", () => {
    render(<AgenticResearchFrontendTools />);
    const addChartSpecTool = registry.find((tool) => tool.name === "add_chart_spec");

    expect(addChartSpecTool).toBeDefined();

    const chartSpecs = [
      { id: "chart-a", title: "Chart A", type: "line", xKey: "x", yKeys: ["y"], data: [{ x: 1, y: 5 }] },
      { id: "chart-b", title: "Chart B", type: "bar", xKey: "x", yKeys: ["y"], data: [{ x: 2, y: 10 }] },
    ];

    const result = addChartSpecTool!.handler({ chartSpecs });

    expect(result.status).toBe("ok");
    expect(result.addedCount).toBe(2);
    expect(result.ids).toEqual(["chart-a", "chart-b"]);
    expect(useAgenticResearchChartStore.getState().chartSpecs).toHaveLength(2);
  });

  // NEW: Test for ar-reorder_chart_specs with valid index mode (lines 184-213)
  it("reorders charts using from_index and to_index", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 1 }], xKey: "x", yKeys: ["y"] },
        { id: "c2", title: "Chart 2", type: "bar", data: [{ x: 2, y: 2 }], xKey: "x", yKeys: ["y"] },
        { id: "c3", title: "Chart 3", type: "area", data: [{ x: 3, y: 3 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();

    // Move chart from index 0 to index 2
    const result = reorderTool!.handler({ from_index: 0, to_index: 2 });

    expect(result.status).toBe("ok");
    expect(result.mode).toBe("index_move");
    expect(result.chart_ids).toEqual(["c2", "c3", "c1"]);
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((s) => s.id)).toEqual(["c2", "c3", "c1"]);
  });

  // NEW: Test for ar-reorder_chart_specs moving chart to earlier position
  it("reorders charts by moving to earlier index", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 1 }], xKey: "x", yKeys: ["y"] },
        { id: "c2", title: "Chart 2", type: "bar", data: [{ x: 2, y: 2 }], xKey: "x", yKeys: ["y"] },
        { id: "c3", title: "Chart 3", type: "area", data: [{ x: 3, y: 3 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();

    // Move chart from index 2 to index 0
    const result = reorderTool!.handler({ from_index: 2, to_index: 0 });

    expect(result.status).toBe("ok");
    expect(result.mode).toBe("index_move");
    expect(result.chart_ids).toEqual(["c3", "c1", "c2"]);
  });

  // NEW: Test for ar-reorder_chart_specs INDEX_OUT_OF_RANGE error (lines 190-203)
  it("returns INDEX_OUT_OF_RANGE when from_index is negative", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 1 }], xKey: "x", yKeys: ["y"] },
        { id: "c2", title: "Chart 2", type: "bar", data: [{ x: 2, y: 2 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();
    const result = reorderTool!.handler({ from_index: -1, to_index: 0 });

    expect(result.status).toBe("error");
    expect(result.code).toBe("INDEX_OUT_OF_RANGE");
    expect(result.from_index).toBe(-1);
    expect(result.to_index).toBe(0);
    expect(result.chart_count).toBe(2);
  });

  // NEW: Test for ar-reorder_chart_specs INDEX_OUT_OF_RANGE when to_index exceeds length
  it("returns INDEX_OUT_OF_RANGE when to_index exceeds chart count", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 1 }], xKey: "x", yKeys: ["y"] },
        { id: "c2", title: "Chart 2", type: "bar", data: [{ x: 2, y: 2 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();
    const result = reorderTool!.handler({ from_index: 0, to_index: 5 });

    expect(result.status).toBe("error");
    expect(result.code).toBe("INDEX_OUT_OF_RANGE");
    expect(result.from_index).toBe(0);
    expect(result.to_index).toBe(5);
    expect(result.chart_count).toBe(2);
  });

  // NEW: Test for ar-reorder_chart_specs INDEX_OUT_OF_RANGE when from_index exceeds length
  it("returns INDEX_OUT_OF_RANGE when from_index exceeds chart count", () => {
    useAgenticResearchChartStore.setState({
      chartSpecs: [
        { id: "c1", title: "Chart 1", type: "line", data: [{ x: 1, y: 1 }], xKey: "x", yKeys: ["y"] },
      ],
    });

    render(<AgenticResearchFrontendTools />);
    const reorderTool = registry.find((tool) => tool.name === "ar-reorder_chart_specs");

    expect(reorderTool).toBeDefined();
    const result = reorderTool!.handler({ from_index: 3, to_index: 0 });

    expect(result.status).toBe("error");
    expect(result.code).toBe("INDEX_OUT_OF_RANGE");
    expect(result.from_index).toBe(3);
    expect(result.to_index).toBe(0);
    expect(result.chart_count).toBe(1);
  });

  // NEW: Test for ar-set_active_dataset INVALID_DATASET_ID error (lines 240-247)
  it("returns INVALID_DATASET_ID when dataset_id is not in manifest", () => {
    render(<AgenticResearchFrontendTools />);
    const setDatasetTool = registry.find((tool) => tool.name === "ar-set_active_dataset");

    expect(setDatasetTool).toBeDefined();
    const result = setDatasetTool!.handler({ dataset_id: "non-existent-dataset" });

    expect(result.status).toBe("error");
    expect(result.code).toBe("INVALID_DATASET_ID");
    expect(result.dataset_id).toBe("non-existent-dataset");
    expect(result.allowed_dataset_ids).toEqual(["dataset-a", "dataset-b"]);
  });

});
