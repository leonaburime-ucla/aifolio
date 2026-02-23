import { describe, expect, it, vi } from "vitest";
import type { ChatCoreStateActions } from "@/features/ai/types/chat.types";
import type { ChartSpec } from "@/features/ai/types/chart.types";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
} from "@/features/ai/orchestrators/chatOrchestrator.helpers";

function createChartSpec(id: string): ChartSpec {
  return {
    id,
    title: `Chart ${id}`,
    type: "line",
    xKey: "x",
    yKeys: ["y"],
    data: [{ x: 1, y: 2 }],
  };
}

function createCoreActions(): ChatCoreStateActions {
  return {
    addMessage: vi.fn(),
    addInputToHistory: vi.fn(),
    moveHistoryCursor: vi.fn(() => ""),
    resetHistoryCursor: vi.fn(),
    setSending: vi.fn(),
    setModelOptions: vi.fn(),
    setSelectedModelId: vi.fn(),
    setModelsLoading: vi.fn(),
  };
}

describe("chatOrchestrator helpers", () => {
  it("maps state with explicit dataset context", () => {
    const mapped = mapChatStateWithDataset(
      {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: false,
      },
      "dataset-123"
    );

    expect(mapped.activeDatasetId).toBe("dataset-123");
  });

  it("maps landing dataset context to null", () => {
    const mapped = mapChatStateWithDataset(
      {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: false,
      },
      null
    );

    expect(mapped.activeDatasetId).toBeNull();
  });

  it("forwards a single chart payload", () => {
    const coreActions = createCoreActions();
    const addChartSpec = vi.fn();
    const actions = composeChatStateActions(coreActions, addChartSpec);
    const chart = createChartSpec("single");

    actions.onMessageReceived({
      message: "ok",
      chartSpec: chart,
    });

    expect(addChartSpec).toHaveBeenCalledTimes(1);
    expect(addChartSpec).toHaveBeenCalledWith(chart);
  });

  it("forwards chart arrays in-order", () => {
    const coreActions = createCoreActions();
    const addChartSpec = vi.fn();
    const actions = composeChatStateActions(coreActions, addChartSpec);
    const chartA = createChartSpec("a");
    const chartB = createChartSpec("b");

    actions.onMessageReceived({
      message: "ok",
      chartSpec: [chartA, chartB],
    });

    expect(addChartSpec).toHaveBeenCalledTimes(2);
    expect(addChartSpec).toHaveBeenNthCalledWith(1, chartA);
    expect(addChartSpec).toHaveBeenNthCalledWith(2, chartB);
  });

  it("ignores null chart payloads", () => {
    const coreActions = createCoreActions();
    const addChartSpec = vi.fn();
    const actions = composeChatStateActions(coreActions, addChartSpec);

    actions.onMessageReceived({
      message: "ok",
      chartSpec: null,
    });

    expect(addChartSpec).not.toHaveBeenCalled();
  });

  it("preserves core action references", () => {
    const coreActions = createCoreActions();
    const addChartSpec = vi.fn();
    const actions = composeChatStateActions(coreActions, addChartSpec);

    expect(actions.addMessage).toBe(coreActions.addMessage);
    expect(actions.setSending).toBe(coreActions.setSending);
    expect(actions.setSelectedModelId).toBe(coreActions.setSelectedModelId);
  });
});
