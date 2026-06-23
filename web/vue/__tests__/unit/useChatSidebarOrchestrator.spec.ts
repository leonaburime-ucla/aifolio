import { describe, it, expect, vi, beforeEach } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { withSetup } from "./helpers/withSetup";
import { useChatSidebarOrchestrator } from "~/features/ai-chat/orchestrator";
import type { ChatApi } from "~/features/ai-chat/api/chatApi";
import { createChatApi } from "~/features/ai-chat/api/chatApi";

vi.mock("~/features/ai-chat/api/chatApi", () => ({
  createChatApi: vi.fn(() => ({
    fetchModels: vi.fn().mockResolvedValue({ models: [], currentModel: null }),
  })),
}));

vi.mock("~/composables/useChartStore", () => ({
  useChartStore: () => ({
    chartSpecs: { value: [] },
    addChartSpec: vi.fn(),
    removeChartSpec: vi.fn(),
    clearCharts: vi.fn(),
  }),
}));

function createMockApi(): ChatApi {
  return {
    fetchModels: vi.fn().mockResolvedValue({
      models: [{ id: "gemini-flash", label: "Gemini Flash" }],
      currentModel: "gemini-flash",
    }),
  };
}

describe("useChatSidebarOrchestrator", () => {
  let api: ChatApi;

  beforeEach(() => {
    api = createMockApi();
  });

  it("loads models on mount", async () => {
    withSetup(() =>
      useChatSidebarOrchestrator({
        baseUrl: "/api/ai",
        getMode: () => "direct",
        getDatasetId: () => null,
        api,
      })
    );

    await flushPromises();

    expect(api.fetchModels).toHaveBeenCalledOnce();
  });

  it("does not expose loadModels to the component", () => {
    const [result] = withSetup(() =>
      useChatSidebarOrchestrator({
        baseUrl: "/api/ai",
        getMode: () => "direct",
        getDatasetId: () => null,
        api,
      })
    );

    expect(result).not.toHaveProperty("loadModels");
  });

  it("populates model options after mount", async () => {
    const [result] = withSetup(() =>
      useChatSidebarOrchestrator({
        baseUrl: "/api/ai",
        getMode: () => "direct",
        getDatasetId: () => null,
        api,
      })
    );

    await flushPromises();

    expect(result.modelOptions.value).toEqual([{ id: "gemini-flash", label: "Gemini Flash" }]);
    expect(result.selectedModelId.value).toBe("gemini-flash");
  });
});
