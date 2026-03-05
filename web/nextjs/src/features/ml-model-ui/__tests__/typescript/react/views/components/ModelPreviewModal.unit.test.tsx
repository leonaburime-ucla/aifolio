import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const useModelPreviewModelMock = vi.fn();

vi.mock("@/features/ml-model-ui/typescript/react/hooks/useModelPreviewModel.hooks", () => ({
  useModelPreviewModel: (params: unknown) => useModelPreviewModelMock(params),
}));

vi.mock("@/core/views/components/General/Modal", () => ({
  Modal: ({
    isOpen,
    onClose,
    title,
    children,
  }: {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: ReactNode;
  }) =>
    isOpen ? (
      <div data-testid="modal">
        <h2>{title}</h2>
        <button type="button" onClick={onClose}>
          close
        </button>
        {children}
      </div>
    ) : null,
}));

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({ children }: { children?: ReactNode }) => (
    <div data-testid="reactflow">{children}</div>
  ),
  Controls: () => <div data-testid="controls" />,
  Background: () => <div data-testid="background" />,
}));

import { ModelPreviewModal } from "@/features/ml-model-ui/typescript/react/views/components/ModelPreviewModal";

describe("ModelPreviewModal", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("renders graph summary, layers, terminology, and passes framework/mode to hook", () => {
    useModelPreviewModelMock.mockReturnValueOnce({
      graph: {
        title: "Wide & Deep",
        summary: "Combined model",
        nodes: [],
        edges: [],
      },
      data: {
        layers: ["Layer A: does thing", "Layer B: does another thing"],
        terminology: [{ term: "Tau", definition: "quantile threshold" }],
      },
    });

    render(
      <ModelPreviewModal
        isOpen
        onClose={vi.fn()}
        framework="tensorflow"
        mode="wide_and_deep"
      />
    );

    expect(useModelPreviewModelMock).toHaveBeenCalledWith({
      framework: "tensorflow",
      mode: "wide_and_deep",
    });
    expect(screen.getByText("Wide & Deep (TensorFlow)")).toBeInTheDocument();
    expect(screen.getByText("Combined model")).toBeInTheDocument();
    expect(screen.getByText("Layer A")).toBeInTheDocument();
    expect(screen.getByText("does thing")).toBeInTheDocument();
    expect(screen.getByText("Key Terminology")).toBeInTheDocument();
    expect(screen.getByText("Tau")).toBeInTheDocument();
    expect(screen.getByTestId("reactflow")).toBeInTheDocument();
  });

  it("renders no-terminology fallback and closes modal", () => {
    const onClose = vi.fn();
    useModelPreviewModelMock.mockReturnValueOnce({
      graph: {
        title: "Linear / GLM Baseline",
        summary: "Simple model",
        nodes: [],
        edges: [],
      },
      data: {
        layers: ["Input Features: tabular data"],
        terminology: [],
      },
    });

    const { container } = render(
      <ModelPreviewModal
        isOpen
        onClose={onClose}
        framework="pytorch"
        mode="linear_glm_baseline"
      />
    );

    const scoped = within(container);
    expect(
      scoped.getByText("No complex terminology explicitly defined for this baseline model.")
    ).toBeInTheDocument();
    fireEvent.click(scoped.getByRole("button", { name: "close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not render when closed", () => {
    useModelPreviewModelMock.mockReturnValueOnce({
      graph: {
        title: "Any",
        summary: "Any",
        nodes: [],
        edges: [],
      },
      data: {
        layers: [],
        terminology: [],
      },
    });

    const { container } = render(
      <ModelPreviewModal
        isOpen={false}
        onClose={vi.fn()}
        framework="pytorch"
        mode="mlp_dense"
      />
    );

    expect(within(container).queryByTestId("modal")).toBeNull();
  });
});
