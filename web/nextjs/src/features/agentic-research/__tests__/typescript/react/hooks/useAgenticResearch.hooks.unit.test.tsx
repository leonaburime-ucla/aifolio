import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  useAgenticResearchIntegration,
  useAgenticResearchLogic,
  useAgenticResearchUiState,
} from "@/features/agentic-research/typescript/react/hooks/useAgenticResearch.hooks";
import type { AgenticResearchDeps } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

function createDeps(overrides: Partial<AgenticResearchDeps> = {}): AgenticResearchDeps {
  const state: AgenticResearchDeps["state"] = {
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
    ...overrides.state,
  };

  const actions: AgenticResearchDeps["actions"] = {
    setDatasetManifest: vi.fn((value) => {
      state.datasetManifest = value;
    }),
    setSelectedDatasetId: vi.fn((value) => {
      state.selectedDatasetId = value;
    }),
    setSklearnTools: vi.fn((value) => {
      state.sklearnTools = value;
    }),
    setTableRows: vi.fn((value) => {
      state.tableRows = value;
    }),
    setTableColumns: vi.fn((value) => {
      state.tableColumns = value;
    }),
    setNumericMatrix: vi.fn((value) => {
      state.numericMatrix = value;
    }),
    setFeatureNames: vi.fn((value) => {
      state.featureNames = value;
    }),
    setPcaChartSpec: vi.fn((value) => {
      state.pcaChartSpec = value;
    }),
    setLoading: vi.fn((value) => {
      state.isLoading = value;
    }),
    setError: vi.fn((value) => {
      state.error = value;
    }),
    ...overrides.actions,
  };

  const api: AgenticResearchDeps["api"] = {
    fetchDatasetManifest: vi.fn(async () => [
      { id: "iris", label: "Iris", description: "iris" },
    ]),
    fetchSklearnTools: vi.fn(async () => ["pca_transform", "linear_regression"]),
    fetchDatasetRows: vi.fn(async () => ({
      rows: [{ "\uFEFF a ": 1, b: 2 }],
      columns: undefined,
    })),
    fetchPcaChartSpec: vi.fn(async () => null),
    ...overrides.api,
  };

  return { state, actions, api };
}

describe("useAgenticResearch.hooks", () => {
  afterEach(() => {
    cleanup();
  });

  it("useAgenticResearchUiState returns placeholder object", () => {
    const { result } = renderHook(() => useAgenticResearchUiState());
    expect(result.current).toEqual({});
  });

  it("loads manifest + tools on mount and allows manual reload", async () => {
    const deps = createDeps();

    const { result } = renderHook(() => useAgenticResearchLogic(deps));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.api.fetchDatasetManifest).toHaveBeenCalled();
    expect(deps.api.fetchSklearnTools).toHaveBeenCalled();
    expect(deps.actions.setDatasetManifest).toHaveBeenCalledWith([
      { id: "iris", label: "Iris", description: "iris" },
    ]);
    expect(deps.actions.setSelectedDatasetId).toHaveBeenCalledWith("iris");

    await act(async () => {
      result.current.reloadManifest();
      await Promise.resolve();
    });

    expect(deps.api.fetchDatasetManifest).toHaveBeenCalledTimes(2);
    act(() => {
      result.current.setSelectedDatasetId("wine");
    });
    expect(deps.actions.setSelectedDatasetId).toHaveBeenCalledWith("wine");
  });

  it("loads selected dataset rows and applies reset + normalization", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris" }],
        selectedDatasetId: "iris",
      },
    });

    renderHook(() => useAgenticResearchLogic(deps));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.api.fetchDatasetRows).toHaveBeenCalledWith("iris");
    expect(deps.actions.setTableColumns).toHaveBeenCalledWith(["\uFEFF a ", "b"]);
    expect(deps.actions.setTableRows).toHaveBeenCalledWith([{ a: 1, b: 2 }]);
    expect(deps.actions.setNumericMatrix).toHaveBeenCalledWith([]);
    expect(deps.actions.setFeatureNames).toHaveBeenCalledWith([]);
    expect(deps.actions.setPcaChartSpec).toHaveBeenCalledWith(null);
  });

  it("returns early when selected id is not present in manifest entries", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "wine", label: "Wine" }],
        selectedDatasetId: "iris",
      },
    });
    renderHook(() => useAgenticResearchLogic(deps));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.api.fetchDatasetRows).not.toHaveBeenCalled();
  });

  it("maps errors from manifest, tools, and dataset loaders", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris" }],
        selectedDatasetId: "iris",
      },
      api: {
        fetchDatasetManifest: vi.fn(async () => {
          throw new Error("manifest boom");
        }),
        fetchSklearnTools: vi.fn(async () => {
          throw "tools boom";
        }),
        fetchDatasetRows: vi.fn(async () => {
          throw "dataset boom";
        }),
      },
    });

    renderHook(() => useAgenticResearchLogic(deps));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.actions.setError).toHaveBeenCalledWith("manifest boom");
    expect(deps.actions.setError).toHaveBeenCalledWith("Failed to load sklearn tools.");
    expect(deps.actions.setError).toHaveBeenCalledWith("Failed to load dataset.");
  });

  it("maps non-Error manifest failures to default copy", async () => {
    const deps = createDeps({
      api: {
        fetchDatasetManifest: vi.fn(async () => {
          throw "bad-manifest";
        }),
      },
    });

    renderHook(() => useAgenticResearchLogic(deps));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.actions.setError).toHaveBeenCalledWith("Failed to load manifest.");
  });

  it("maps Error instances for tools and dataset loaders", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris" }],
        selectedDatasetId: "iris",
      },
      api: {
        fetchSklearnTools: vi.fn(async () => {
          throw new Error("tools exploded");
        }),
        fetchDatasetRows: vi.fn(async () => {
          throw new Error("dataset exploded");
        }),
      },
    });

    renderHook(() => useAgenticResearchLogic(deps));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.actions.setError).toHaveBeenCalledWith("tools exploded");
    expect(deps.actions.setError).toHaveBeenCalledWith("dataset exploded");
  });

  it("uses empty rows fallback when dataset payload omits rows", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris" }],
        selectedDatasetId: "iris",
      },
      api: {
        fetchDatasetRows: vi.fn(async () => ({
          rows: undefined,
          columns: ["a"],
        })),
      },
    });

    renderHook(() => useAgenticResearchLogic(deps));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(deps.actions.setTableColumns).toHaveBeenCalledWith(["a"]);
    expect(deps.actions.setTableRows).toHaveBeenCalledWith([]);
  });

  it("skips dataset loading when selection is missing or manifest is empty", async () => {
    const noSelectionDeps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris" }],
        selectedDatasetId: null,
      },
    });
    renderHook(() => useAgenticResearchLogic(noSelectionDeps));

    const emptyManifestDeps = createDeps({
      state: {
        datasetManifest: [],
        selectedDatasetId: "iris",
      },
    });
    renderHook(() => useAgenticResearchLogic(emptyManifestDeps));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(noSelectionDeps.api.fetchDatasetRows).not.toHaveBeenCalled();
    expect(emptyManifestDeps.api.fetchDatasetRows).not.toHaveBeenCalled();
  });

  it("integration exposes groupedTools + datasetOptions", async () => {
    const deps = createDeps({
      state: {
        datasetManifest: [{ id: "iris", label: "Iris", description: "d" }],
        selectedDatasetId: "iris",
        sklearnTools: ["linear_regression", "pca_transform"],
      },
    });

    const { result } = renderHook(() => useAgenticResearchIntegration(deps));

    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.datasetOptions).toEqual([
      { id: "iris", label: "Iris", description: "d" },
    ]);
    expect(result.current.groupedTools.Regression).toEqual(["linear_regression"]);
    expect(result.current.groupedTools["Decomposition & Embeddings"]).toEqual([
      "pca_transform",
    ]);
    expect(typeof result.current.reloadManifest).toBe("function");
  });
});
