import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import {
  useAgenticResearchIntegration,
  useAgenticResearchLogic,
  useAgenticResearchUiState,
} from "@/features/agentic-research/hooks/useAgenticResearch.hooks";
import type { AgenticResearchDeps } from "@/features/agentic-research/types/agenticResearch.types";

function createActions() {
  return {
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
  };
}

describe("useAgenticResearchLogic", () => {
  it("applies manifest fallback selection when selected dataset is null", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [
          { id: "dataset-a", label: "Dataset A" },
          { id: "dataset-b", label: "Dataset B" },
        ]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setDatasetManifest).toHaveBeenCalled();
      expect(actions.setSelectedDatasetId).toHaveBeenCalledWith("dataset-a");
    });
  });

  it("clears stale table/chart numeric state before applying dataset rows", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-a",
        sklearnTools: [],
        tableRows: [{ stale: 1 }],
        tableColumns: ["stale"],
        numericMatrix: [[1]],
        featureNames: ["stale"],
        pcaChartSpec: {
          id: "stale-chart",
          title: "stale",
          type: "line",
          data: [{ x: 1, y: 2 }],
          xKey: "x",
          yKeys: ["y"],
        },
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({
          rows: [{ col_a: 10, col_b: 20 }],
          columns: ["col_a", "col_b"],
        })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(deps.api.fetchDatasetRows).toHaveBeenCalledWith("dataset-a");
      expect(actions.setTableRows).toHaveBeenCalledWith([]);
      expect(actions.setTableColumns).toHaveBeenCalledWith([]);
      expect(actions.setPcaChartSpec).toHaveBeenCalledWith(null);
      expect(actions.setNumericMatrix).toHaveBeenCalledWith([]);
      expect(actions.setFeatureNames).toHaveBeenCalledWith([]);
      expect(actions.setTableColumns).toHaveBeenCalledWith(["col_a", "col_b"]);
      expect(actions.setTableRows).toHaveBeenCalledWith([{ col_a: 10, col_b: 20 }]);
    });
  });

  it("keeps existing selected dataset id when manifest loads", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: "dataset-existing",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setSelectedDatasetId).toHaveBeenCalledWith("dataset-existing");
    });
  });

  it("sets sklearn tools error state when tool fetch fails", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => []),
        fetchSklearnTools: vi.fn(async () => {
          throw new Error("tools failed");
        }),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setSklearnTools).toHaveBeenCalledWith([]);
      expect(actions.setError).toHaveBeenCalledWith("tools failed");
    });
  });

  it("sets manifest fallback error message when manifest throws non-Error", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => {
          throw "manifest-failed";
        }),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setError).toHaveBeenCalledWith("Failed to load manifest.");
    });
  });

  it("uses manifest error message when manifest throws Error instance", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => {
          throw new Error("manifest failed as error");
        }),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setError).toHaveBeenCalledWith("manifest failed as error");
    });
  });

  it("does not call dataset rows API when selected dataset is missing", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(deps.api.fetchDatasetRows).not.toHaveBeenCalled();
    });
  });

  it("sets dataset fallback error message when dataset load throws non-Error", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-a",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => {
          throw "dataset-failed";
        }),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setError).toHaveBeenCalledWith("Failed to load dataset.");
      expect(actions.setTableRows).toHaveBeenCalledWith([]);
      expect(actions.setTableColumns).toHaveBeenCalledWith([]);
      expect(actions.setNumericMatrix).toHaveBeenCalledWith([]);
      expect(actions.setFeatureNames).toHaveBeenCalledWith([]);
      expect(actions.setPcaChartSpec).toHaveBeenCalledWith(null);
    });
  });

  it("sets sklearn fallback error message when tools throws non-Error", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [],
        selectedDatasetId: null,
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => []),
        fetchSklearnTools: vi.fn(async () => {
          throw "tools non error";
        }),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setSklearnTools).toHaveBeenCalledWith([]);
      expect(actions.setError).toHaveBeenCalledWith("Failed to load sklearn tools.");
    });
  });

  it("does not load dataset rows when selected dataset id is not found in manifest", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-missing",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(deps.api.fetchDatasetRows).not.toHaveBeenCalled();
    });
  });

  it("derives columns from rows when dataset payload omits columns", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-a",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({
          rows: [{ c1: 1, c2: 2 }],
        })),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setTableColumns).toHaveBeenCalledWith(["c1", "c2"]);
    });
  });

  it("defaults rows to empty array when dataset payload omits rows", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-a",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => ({})),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setTableRows).toHaveBeenCalledWith([]);
      expect(actions.setTableColumns).toHaveBeenCalledWith([]);
    });
  });

  it("uses dataset error message when dataset load throws Error instance", async () => {
    const actions = createActions();
    const deps = {
      state: {
        datasetManifest: [{ id: "dataset-a", label: "Dataset A" }],
        selectedDatasetId: "dataset-a",
        sklearnTools: [],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions,
      api: {
        fetchDatasetManifest: vi.fn(async () => [{ id: "dataset-a", label: "Dataset A" }]),
        fetchSklearnTools: vi.fn(async () => []),
        fetchDatasetRows: vi.fn(async () => {
          throw new Error("rows failed");
        }),
        fetchPcaChartSpec: vi.fn(async () => null),
      },
    } satisfies AgenticResearchDeps;

    renderHook(() => useAgenticResearchLogic(deps));

    await waitFor(() => {
      expect(actions.setError).toHaveBeenCalledWith("rows failed");
    });
  });

  it("useAgenticResearchUiState returns an empty ui state object", () => {
    const { result } = renderHook(() => useAgenticResearchUiState());
    expect(result.current).toEqual({});
  });

  it("integration groups tools across all categories", () => {
    const deps = {
      state: {
        datasetManifest: [{ id: "d1", label: "Dataset 1", description: "desc" }],
        selectedDatasetId: "d1",
        sklearnTools: [
          "linear_regression",
          "binary_classification",
          "kmeans_clustering",
          "pca_transform",
          "standard_scaler",
          "select_k_best",
          "train_test_split",
          "roc_auc_score",
          "custom_tool",
        ],
        tableRows: [],
        tableColumns: [],
        numericMatrix: [],
        featureNames: [],
        pcaChartSpec: null,
        isLoading: false,
        error: null,
      },
      actions: createActions(),
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
    expect(result.current.groupedTools.Regression).toEqual(["linear_regression"]);
    expect(result.current.groupedTools.Classification).toEqual(["binary_classification"]);
    expect(result.current.groupedTools.Clustering).toEqual(["kmeans_clustering"]);
    expect(result.current.groupedTools["Decomposition & Embeddings"]).toEqual([
      "pca_transform",
    ]);
    expect(result.current.groupedTools.Preprocessing).toEqual(["standard_scaler"]);
    expect(result.current.groupedTools["Feature Selection"]).toEqual(["select_k_best"]);
    expect(result.current.groupedTools["Model Selection"]).toEqual(["train_test_split"]);
    expect(result.current.groupedTools.Metrics).toEqual(["roc_auc_score"]);
    expect(result.current.groupedTools.Other).toEqual(["custom_tool"]);
  });
});
