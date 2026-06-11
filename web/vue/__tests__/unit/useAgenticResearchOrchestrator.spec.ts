import { describe, it, expect, vi, beforeEach } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { withSetup } from "./helpers/withSetup";
import { useAgenticResearchOrchestrator } from "~/features/agentic-research/orchestrator";
import type { AgenticResearchApi } from "~/features/agentic-research/model";
import { createAgenticResearchApi } from "~/features/agentic-research/api";

vi.mock("~/features/agentic-research/api", () => ({
  createAgenticResearchApi: vi.fn(() => ({
    loadManifest: vi.fn().mockResolvedValue([]),
    loadTools: vi.fn().mockResolvedValue([]),
    loadDataset: vi.fn().mockResolvedValue({ rows: [], columns: [] }),
  })),
}));

function createMockApi(overrides: Partial<AgenticResearchApi> = {}): AgenticResearchApi {
  return {
    loadManifest: vi.fn().mockResolvedValue([
      { id: "churn.csv", label: "churn.csv", description: "Customer churn" },
    ]),
    loadTools: vi.fn().mockResolvedValue(["pca_transform"]),
    loadDataset: vi.fn().mockResolvedValue({ rows: [{ a: 1 }], columns: ["a"] }),
    ...overrides,
  };
}

describe("useAgenticResearchOrchestrator", () => {
  let api: AgenticResearchApi;

  beforeEach(() => {
    api = createMockApi();
  });

  it("calls init on mount", async () => {
    const [result] = withSetup(() =>
      useAgenticResearchOrchestrator({ baseUrl: "/api/ai", api })
    );

    await flushPromises();

    expect(api.loadManifest).toHaveBeenCalledOnce();
    expect(api.loadTools).toHaveBeenCalledOnce();
  });

  it("does not expose init or onDatasetWatch to the component", () => {
    const [result] = withSetup(() =>
      useAgenticResearchOrchestrator({ baseUrl: "/api/ai", api })
    );

    expect(result).not.toHaveProperty("init");
    expect(result).not.toHaveProperty("onDatasetWatch");
  });

  it("watches selectedDatasetId and loads dataset on change", async () => {
    const [result] = withSetup(() =>
      useAgenticResearchOrchestrator({ baseUrl: "/api/ai", api })
    );

    await flushPromises();

    result.selectedDatasetId.value = "churn.csv";
    await flushPromises();

    expect(api.loadDataset).toHaveBeenCalledWith("churn.csv");
  });

  it("fires onDatasetChange callback", async () => {
    const spy = vi.fn();
    const [result] = withSetup(() =>
      useAgenticResearchOrchestrator({ baseUrl: "/api/ai", api, onDatasetChange: spy })
    );

    await flushPromises();

    result.onDatasetChange("fraud.csv");
    expect(spy).toHaveBeenCalledWith("fraud.csv");
  });
});
