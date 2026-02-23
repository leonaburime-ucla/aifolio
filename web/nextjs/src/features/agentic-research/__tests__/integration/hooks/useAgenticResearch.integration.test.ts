import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgenticResearchIntegration } from "@/features/agentic-research/hooks/useAgenticResearch.hooks";
import type { AgenticResearchDeps } from "@/features/agentic-research/types/agenticResearch.types";

describe("useAgenticResearchIntegration", () => {
  it("maps dataset options and groups sklearn tools", () => {
    const deps = {
      state: {
        datasetManifest: [
          { id: "d1", label: "Dataset 1", description: "desc" },
        ],
        selectedDatasetId: "d1",
        sklearnTools: ["pca_transform", "logistic_classification", "kmeans_clustering"],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions: {
        setDatasetManifest: vi.fn(),
        setSelectedDatasetId: vi.fn(),
        setSklearnTools: vi.fn(),
        setTableRows: vi.fn(),
        setTableColumns: vi.fn(),
        setNumericMatrix: vi.fn(),
        setFeatureNames: vi.fn(),
        setPcaChartSpec: vi.fn(),
        setLoading: vi.fn(),
        setError: vi.fn(),
      },
      api: {
        fetchDatasetManifest: vi.fn(async () => []),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    const { result } = renderHook(() => useAgenticResearchIntegration(deps));

    expect(result.current.datasetOptions).toEqual([
      { id: "d1", label: "Dataset 1", description: "desc" },
    ]);
    expect(result.current.groupedTools["Decomposition & Embeddings"]).toEqual(["pca_transform"]);
    expect(result.current.groupedTools.Classification).toEqual(["logistic_classification"]);
    expect(result.current.groupedTools.Clustering).toEqual(["kmeans_clustering"]);
  });
});
