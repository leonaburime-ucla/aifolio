import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useModelPreviewModel } from "@/features/ml-model-ui/typescript/react/hooks/useModelPreviewModel.hooks";

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
    const { result, rerender } = renderHook(
      ({ framework, mode }: { framework: "pytorch" | "tensorflow"; mode: "mlp_dense" | "tabresnet" }) =>
        useModelPreviewModel({ framework, mode }),
      {
        initialProps: { framework: "tensorflow" as const, mode: "mlp_dense" as const },
      }
    );

    const initial = result.current;
    rerender({ framework: "tensorflow", mode: "mlp_dense" });
    expect(result.current).toBe(initial);

    rerender({ framework: "tensorflow", mode: "tabresnet" });
    expect(result.current).not.toBe(initial);
    expect(result.current.graph.title).toContain("TabResNet");
  });
});
