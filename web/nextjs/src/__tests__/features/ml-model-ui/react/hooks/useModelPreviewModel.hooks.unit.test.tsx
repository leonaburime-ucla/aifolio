import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useModelPreviewModel } from "@/features/ml-model-ui/react/hooks/useModelPreviewModel.hooks";
import type {
  ModelPreviewFramework,
  ModelPreviewMode,
} from "@/features/ml-model-ui/react/views/components/ModelPreviewModal.types";

describe("useModelPreviewModel", () => {
  it("returns graph/bullets and normalizes node text color", () => {
    const { result } = renderHook(() =>
      useModelPreviewModel({ framework: "pytorch", mode: "mlp_dense" })
    );

    expect(result.current.graph.title).toContain("Perceptron");
    expect(result.current.data.layers.length).toBeGreaterThan(0);
    expect(
      result.current.graph.nodes.every((node) => node.style?.color === "#18181b")
    ).toBe(true);
  });

  it("memoizes values by framework/mode and recomputes on change", () => {
    const initialProps: {
      framework: ModelPreviewFramework;
      mode: ModelPreviewMode;
    } = {
      framework: "tensorflow",
      mode: "mlp_dense",
    };

    const { result, rerender } = renderHook(
      ({ framework, mode }: { framework: ModelPreviewFramework; mode: ModelPreviewMode }) =>
        useModelPreviewModel({ framework, mode }),
      { initialProps }
    );

    const initial = result.current;
    rerender({ framework: "tensorflow", mode: "mlp_dense" });
    expect(result.current).toBe(initial);

    rerender({ framework: "tensorflow", mode: "tabresnet" });
    expect(result.current).not.toBe(initial);
    expect(result.current.graph.title).toContain("TabResNet");
  });
});
