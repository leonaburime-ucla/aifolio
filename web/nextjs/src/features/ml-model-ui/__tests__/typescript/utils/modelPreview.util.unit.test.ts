import { describe, expect, it } from "vitest";
import {
  buildModelPreviewBullets,
  buildModelPreviewGraph,
  parseLayerBullet,
} from "@/features/ml-model-ui/typescript/utils/modelPreview.util";
import type { ModelPreviewMode } from "@/features/ml-model-ui/__types__/typescript/react/views/modelPreviewModal.types";

const MODES: ModelPreviewMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "tabresnet",
  "wide_and_deep",
  "imbalance_aware",
  "quantile_regression",
  "calibrated_classifier",
  "entity_embeddings",
  "autoencoder_head",
  "multi_task_learning",
  "time_aware_tabular",
  "tree_teacher_distillation",
];

describe("modelPreview.util", () => {
  it("builds graph models for all supported modes", () => {
    MODES.forEach((mode) => {
      const graph = buildModelPreviewGraph({ framework: "pytorch", mode });
      expect(graph.title.length).toBeGreaterThan(0);
      expect(graph.summary.length).toBeGreaterThan(0);
      expect(graph.nodes.length).toBeGreaterThan(0);
      expect(graph.edges.length).toBeGreaterThan(0);
      expect(graph.nodes.some((node) => node.id === "input")).toBe(true);
    });
  });

  it("builds bullets for all supported modes", () => {
    MODES.forEach((mode) => {
      const bullets = buildModelPreviewBullets({ mode });
      expect(bullets.layers.length).toBeGreaterThan(0);
      expect(bullets.terminology.length).toBeGreaterThan(0);
    });
  });

  it("uses quantile-specific output labels", () => {
    const quantile = buildModelPreviewGraph({
      framework: "tensorflow",
      mode: "quantile_regression",
    });
    expect(quantile.nodes.some((node) => String(node.data?.label).includes("P80"))).toBe(true);
  });

  it("uses fallback graph and bullet defaults for unknown mode", () => {
    const unknownMode = "unknown_mode" as ModelPreviewMode;
    const fallbackTorch = buildModelPreviewGraph({
      framework: "pytorch",
      mode: unknownMode,
    });
    expect(fallbackTorch.title).toBe("PyTorch Neural Net");
    expect(fallbackTorch.edges).toEqual([{ id: "fallback", source: "input", target: "output" }]);

    const fallbackTf = buildModelPreviewGraph({
      framework: "tensorflow",
      mode: unknownMode,
    });
    expect(fallbackTf.title).toBe("TensorFlow Neural Net");

    const fallbackBullets = buildModelPreviewBullets({ mode: unknownMode });
    expect(fallbackBullets.layers[0]).toContain("Input Features");
    expect(fallbackBullets.terminology[0]?.term).toBe("Features");
  });

  it("parses layer bullets with and without a separator", () => {
    expect(parseLayerBullet("Dense Block: learns interactions")).toEqual({
      term: "Dense Block",
      definition: "learns interactions",
    });
    expect(parseLayerBullet("Dense Block only")).toEqual({
      term: "Dense Block only",
      definition: "",
    });
    expect(parseLayerBullet("Quantile Head: tau:0.8 target")).toEqual({
      term: "Quantile Head",
      definition: "tau:0.8 target",
    });
  });
});
