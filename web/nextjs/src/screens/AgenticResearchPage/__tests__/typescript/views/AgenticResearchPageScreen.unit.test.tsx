import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgenticResearchOrchestratorModel } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

vi.mock("next/dynamic", () => ({
  default: () => () => null,
}));

vi.mock("@/features/recharts/typescript/react/views/components/ChartRenderer", () => ({
  default: ({
    spec,
  }: {
    spec: { id: string; title: string };
  }) => <div data-testid="chart-renderer">{`${spec.id}:${spec.title}`}</div>,
}));

vi.mock("@/core/views/components/Datatable/DataTable", () => ({
  default: () => <div data-testid="data-table" />,
}));

vi.mock("@/features/agentic-research/typescript/react/views/components/DatasetCombobox", () => ({
  default: () => <div data-testid="dataset-combobox" />,
}));

import AgenticResearchPageScreen from "@/screens/AgenticResearchPage/views/AgenticResearchPageScreen";

afterEach(() => {
  cleanup();
});

const chart = (id: string, title = id) => ({
  id,
  title,
  type: "scatter" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
});

function createModel(
  overrides: Partial<AgenticResearchOrchestratorModel> = {}
): AgenticResearchOrchestratorModel {
  return {
    datasetManifest: [{ id: "fraud_detection", label: "Fraud Detection" }],
    selectedDatasetId: "fraud_detection",
    sklearnTools: ["pca_transform", "nmf_decomposition", "plsr_regression", "lasso_regression"],
    tableRows: [],
    tableColumns: [],
    numericMatrix: [],
    featureNames: [],
    pcaChartSpec: null,
    isLoading: false,
    error: null,
    groupedTools: {
      "Decomposition & Embeddings": ["pca_transform", "nmf_decomposition"],
      Regression: ["plsr_regression", "lasso_regression"],
    },
    datasetOptions: [{ id: "fraud_detection", label: "Fraud Detection" }],
    reloadManifest: vi.fn(),
    setSelectedDatasetId: vi.fn(),
    activeChartSpec: null,
    chartSpecs: [],
    removeChartSpec: vi.fn(),
    formatToolName: (name: string) => name,
    ...overrides,
  };
}

describe("AgenticResearchPageScreen", () => {
  it("renders stacked charts in newest-first order and enables scroll when more than two exist", () => {
    const model = createModel({
      chartSpecs: [
        chart("lasso", "Lasso Regression"),
        chart("nmf", "NMF Decomposition"),
        chart("plsr", "PLSR"),
        chart("pca", "PCA Projection"),
      ],
    });

    const { container } = render(
      <AgenticResearchPageScreen
        showChatSidebar={false}
        pageOrchestrator={() => model}
      />
    );

    const scrollContainer = container.querySelector(".overflow-y-auto");
    expect(scrollContainer).not.toBeNull();

    const chartNodes = within(scrollContainer as HTMLElement).getAllByTestId("chart-renderer");
    expect(chartNodes.map((node) => node.textContent)).toEqual([
      "lasso:Lasso Regression",
      "nmf:NMF Decomposition",
      "plsr:PLSR",
      "pca:PCA Projection",
    ]);
  });

  it("falls back to the single active chart renderer when chartSpecs is empty", () => {
    const model = createModel({
      activeChartSpec: chart("pca", "PCA Projection"),
    });

    render(
      <AgenticResearchPageScreen
        showChatSidebar={false}
        pageOrchestrator={() => model}
      />
    );

    expect(screen.getByTestId("chart-renderer")).toHaveTextContent("pca:PCA Projection");
  });
});
