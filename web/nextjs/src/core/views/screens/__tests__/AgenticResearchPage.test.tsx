import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ChartSpec } from "@/features/ai/types/chart.types";
import AgenticResearchPage from "@/core/views/screens/AgenticResearchPage";

const chatSidebarSpy = vi.fn();

vi.mock("@/features/ai/views/components/ChatSidebar", () => ({
  default: (props: { chatOrchestrator: unknown }) => {
    chatSidebarSpy(props);
    return <div data-testid="chat-sidebar" />;
  },
}));

vi.mock("@/features/recharts/views/components/ChartRenderer", () => ({
  default: ({ spec }: { spec: ChartSpec }) => (
    <div data-testid="chart-renderer">{spec.id}</div>
  ),
}));

vi.mock("@/core/views/components/Datatable/DataTable", () => ({
  default: ({ rows, columns }: { rows: unknown[]; columns: string[] }) => (
    <div data-testid="data-table">{`rows:${rows.length};cols:${columns.length}`}</div>
  ),
}));

vi.mock("@/features/agentic-research/views/components/DatasetCombobox", () => ({
  default: () => <div data-testid="dataset-combobox" />,
}));

describe("AgenticResearchPage", () => {
  it("uses injected page and chat orchestrators", () => {
    const pageOrchestrator = vi.fn(() => ({
      isLoading: false,
      error: null,
      datasetOptions: [],
      selectedDatasetId: null,
      setSelectedDatasetId: vi.fn(),
      sklearnTools: ["pca_transform"],
      tableRows: [{ a: 1 }],
      tableColumns: ["a"],
      activeChartSpec: null,
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
      groupedTools: { "Decomposition & Embeddings": ["pca_transform"] },
      formatToolName: (name: string) => name,
      datasetManifest: [],
      numericMatrix: [],
      featureNames: [],
      pcaChartSpec: null,
      reloadManifest: vi.fn(),
    }));

    const chatOrchestrator = vi.fn();

    render(
      <AgenticResearchPage
        pageOrchestrator={pageOrchestrator}
        chatOrchestrator={chatOrchestrator as never}
      />
    );

    expect(pageOrchestrator).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("chat-sidebar")).toBeInTheDocument();
    expect(screen.getByTestId("data-table")).toHaveTextContent("rows:1;cols:1");
    expect(chatSidebarSpy).toHaveBeenCalled();
    expect(chatSidebarSpy.mock.calls[0][0].chatOrchestrator).toBe(chatOrchestrator);
  });
});
