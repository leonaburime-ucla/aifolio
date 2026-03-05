import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

const useChartRendererModelMock = vi.fn();

vi.mock("@/features/recharts/typescript/react/hooks/useChartRendererModel.hooks", () => ({
  useChartRendererModel: (params: unknown) => useChartRendererModelMock(params),
}));

vi.mock("@/features/recharts/typescript/react/views/components/EChartsRenderer", () => ({
  default: () => <div data-testid="echarts-renderer" />,
}));

vi.mock("@/core/views/components/General/Modal", () => ({
  Modal: ({
    isOpen,
    onClose,
    children,
  }: {
    isOpen: boolean;
    onClose?: () => void;
    children: ReactNode;
  }) =>
    isOpen ? (
      <div data-testid="modal">
        <button type="button" data-testid="modal-close" onClick={onClose}>
          close
        </button>
        {children}
      </div>
    ) : null,
}));

vi.mock("recharts", () => {
  const make = (name: string) =>
    ({ children }: { children?: ReactNode }) => (
      <div data-testid={name}>{children}</div>
    );
  const XAxis = ({
    children,
    tickFormatter,
  }: {
    children?: ReactNode;
    tickFormatter?: (value: unknown) => unknown;
  }) => {
    tickFormatter?.("x");
    return <div data-testid="XAxis">{children}</div>;
  };
  const YAxis = ({
    children,
    tickFormatter,
  }: {
    children?: ReactNode;
    tickFormatter?: (value: unknown) => unknown;
  }) => {
    tickFormatter?.(1);
    return <div data-testid="YAxis">{children}</div>;
  };
  const Tooltip = ({
    children,
    formatter,
    content,
  }: {
    children?: ReactNode;
    formatter?: (value: unknown) => unknown;
    content?: (args: unknown) => ReactNode;
  }) => {
    formatter?.(2);
    content?.({
      active: true,
      payload: [
        {
          dataKey: "y",
          name: "Y",
          value: 2,
          payload: { feature: "feature-a" },
        },
      ],
    });
    content?.({
      active: true,
      payload: [
        {
          dataKey: "y",
          name: "Y",
          value: 3,
          payload: {},
        },
      ],
    });
    content?.({
      active: true,
      payload: [
        {
          dataKey: "y",
          value: 4,
          payload: { feature: "feature-b" },
        },
      ],
    });
    content?.({ active: false, payload: [] });
    return <div data-testid="Tooltip">{children}</div>;
  };
  return {
    ResponsiveContainer: make("ResponsiveContainer"),
    AreaChart: make("AreaChart"),
    BarChart: make("BarChart"),
    LineChart: make("LineChart"),
    ScatterChart: make("ScatterChart"),
    XAxis,
    YAxis,
    Tooltip,
    Legend: make("Legend"),
    Area: make("Area"),
    Bar: make("Bar"),
    Line: make("Line"),
    Scatter: make("Scatter"),
    LabelList: make("LabelList"),
    ErrorBar: make("ErrorBar"),
  };
});

import ChartRenderer, {
  LoadingLabel,
  renderUnsupportedChart,
} from "@/features/recharts/typescript/react/views/components/ChartRenderer";

const baseModel = {
  isExpanded: false,
  setIsExpanded: vi.fn(),
  yKeys: ["y"],
  scatterLabelKey: null,
  data: [{ x: 1, y: 2 }],
  chartProps: {
    data: [{ x: 1, y: 2 }],
    margin: { top: 12, right: 20, left: 28, bottom: 40 },
  },
};

const baseSpec = {
  id: "chart-1",
  title: "Chart 1",
  type: "line" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [{ x: 1, y: 2 }],
};

describe("ChartRenderer", () => {
  it("LoadingLabel renders only when coordinates/value are present", () => {
    expect(LoadingLabel({ x: undefined, y: 2, value: "a" })).toBeNull();
    const rendered = LoadingLabel({ x: 1, y: 2, value: "a" });
    expect(rendered).toBeTruthy();
  });

  it("renders unsupported chart helper copy", () => {
    const { container } = render(renderUnsupportedChart({ ...baseSpec, type: "violin" }));
    expect(container.textContent).toContain("Unsupported chart type: violin");
  });

  it("renders line chart branch and remove button", () => {
    const onRemove = vi.fn();
    const setIsExpanded = vi.fn();
    useChartRendererModelMock.mockReturnValueOnce({ ...baseModel, setIsExpanded });

    render(<ChartRenderer spec={baseSpec} onRemove={onRemove} />);

    expect(screen.getByText("Chart 1")).toBeInTheDocument();
    expect(screen.getByTestId("LineChart")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Remove chart Chart 1"));
    expect(onRemove).toHaveBeenCalledWith("chart-1");
    fireEvent.click(screen.getByLabelText("Expand chart Chart 1"));
    expect(setIsExpanded).toHaveBeenCalledWith(true);
  });

  it("renders heatmap branch through EChartsRenderer", () => {
    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    render(<ChartRenderer spec={{ ...baseSpec, type: "heatmap" }} />);
    expect(screen.getByTestId("echarts-renderer")).toBeInTheDocument();
  });

  it("renders unsupported violin/surface branch and metadata", () => {
    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    render(
      <ChartRenderer
        spec={{
          ...baseSpec,
          type: "surface",
          meta: { datasetLabel: "iris", queryTimeMs: 1200 },
        }}
      />
    );

    expect(screen.getByText("Unsupported chart type: surface")).toBeInTheDocument();
    expect(screen.getByText("Dataset: iris")).toBeInTheDocument();
    expect(screen.getByText("Query time: 1.20s")).toBeInTheDocument();
  });

  it("renders scatter/bar/area/errorbar branches", () => {
    useChartRendererModelMock.mockReturnValueOnce({
      ...baseModel,
      scatterLabelKey: "feature",
      yKeys: ["y", "z"],
    });
    const { rerender } = render(
      <ChartRenderer spec={{ ...baseSpec, type: "scatter", yKeys: ["y", "z"] }} />
    );
    expect(screen.getByTestId("ScatterChart")).toBeInTheDocument();
    expect(screen.getByTestId("Legend")).toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "bar" }} />);
    expect(screen.getByTestId("BarChart")).toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "density" }} />);
    expect(screen.getByTestId("AreaChart")).toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(
      <ChartRenderer
        spec={{ ...baseSpec, type: "errorbar", errorKeys: { y: "y_err" } }}
      />
    );
    expect(screen.getAllByTestId("LineChart").length).toBeGreaterThan(0);
    expect(screen.getByTestId("ErrorBar")).toBeInTheDocument();
  });

  it("renders expanded modal content with description and meta", () => {
    const setIsExpanded = vi.fn();
    useChartRendererModelMock.mockReturnValueOnce({
      ...baseModel,
      isExpanded: true,
      setIsExpanded,
    });

    render(
      <ChartRenderer
        spec={{
          ...baseSpec,
          description: "Detailed chart",
          xLabel: "X axis",
          yLabel: "Y axis",
          meta: { datasetLabel: "housing", queryTimeMs: 2500 },
        }}
      />
    );

    expect(screen.getByTestId("modal")).toBeInTheDocument();
    expect(screen.getAllByText("Detailed chart").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Dataset: housing").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Query time: 2.50s").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByTestId("modal-close"));
    expect(setIsExpanded).toHaveBeenCalledWith(false);
  });

  it("covers additional chart type and metadata edge branches", () => {
    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    const { container, rerender } = render(<ChartRenderer spec={{ ...baseSpec, type: "box" }} />);
    expect(within(container).getAllByTestId("echarts-renderer").length).toBeGreaterThan(0);

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "histogram" }} />);
    expect(within(container).getByTestId("BarChart")).toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "roc" }} />);
    expect(within(container).getAllByTestId("LineChart").length).toBeGreaterThan(0);

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(
      <ChartRenderer spec={{ ...baseSpec, type: "errorbar", errorKeys: { z: "z_err" } }} />
    );
    expect(within(container).queryByTestId("ErrorBar")).not.toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce({
      ...baseModel,
      isExpanded: true,
    });
    rerender(
      <ChartRenderer
        spec={{
          ...baseSpec,
          description: "Dataset only",
          meta: { datasetLabel: "iris-only" },
        }}
      />
    );
    expect(within(container).getAllByText("Dataset: iris-only").length).toBeGreaterThan(0);
    expect(within(container).queryByText(/Query time:/)).not.toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "biplot" }} />);
    expect(within(container).getByTestId("ScatterChart")).toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce({ ...baseModel, scatterLabelKey: null });
    rerender(<ChartRenderer spec={{ ...baseSpec, type: "scatter" }} />);
    expect(within(container).queryByTestId("LabelList")).not.toBeInTheDocument();

    useChartRendererModelMock.mockReturnValueOnce(baseModel);
    rerender(
      <ChartRenderer
        spec={{
          ...baseSpec,
          meta: { queryTimeMs: 300 },
        }}
      />
    );
    expect(within(container).getAllByText("Query time: 0.30s").length).toBeGreaterThan(0);
    expect(within(container).queryByText("Dataset:")).not.toBeInTheDocument();
  });
});
