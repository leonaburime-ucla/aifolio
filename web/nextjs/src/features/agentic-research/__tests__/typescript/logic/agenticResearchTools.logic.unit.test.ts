import { describe, expect, it } from "vitest";
import { groupSklearnTools } from "@/features/agentic-research/typescript/logic/agenticResearchTools.logic";

describe("agenticResearchTools.logic", () => {
  it("groups sklearn tools into deterministic categories", () => {
    const grouped = groupSklearnTools({
      tools: [
        "linear_regression",
        "binary_classification",
        "kmeans_clustering",
        "pca_transform",
        "standard_scaler",
        "select_k_best",
        "train_test_split",
        "roc_auc_score",
        "custom_helper",
      ],
    });

    expect(grouped.Regression).toEqual(["linear_regression"]);
    expect(grouped.Classification).toEqual(["binary_classification"]);
    expect(grouped.Clustering).toEqual(["kmeans_clustering"]);
    expect(grouped["Decomposition & Embeddings"]).toEqual(["pca_transform"]);
    expect(grouped.Preprocessing).toEqual(["standard_scaler"]);
    expect(grouped["Feature Selection"]).toEqual(["select_k_best"]);
    expect(grouped["Model Selection"]).toEqual(["train_test_split"]);
    expect(grouped.Metrics).toEqual(["roc_auc_score"]);
    expect(grouped.Other).toEqual(["custom_helper"]);
  });
});
