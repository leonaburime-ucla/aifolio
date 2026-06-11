import { describe, it, expect, vi } from "vitest";
import {
  mapChatStateWithDataset,
  createOnMessageReceived,
  composeChatStateActions,
} from "../../../src/chat/index";
import type { ChatCoreStateActions } from "@aifolio/contracts/entities/chat";

describe("mapChatStateWithDataset", () => {
  it("merges activeDatasetId into state", () => {
    const state = {
      messages: [],
      inputHistory: [],
      historyCursor: null,
      isSending: false,
      modelOptions: [],
      selectedModelId: null,
      isModelsLoading: false,
      screenFeedback: null,
    };
    const result = mapChatStateWithDataset({ state, activeDatasetId: "ds-1" });
    expect(result.activeDatasetId).toBe("ds-1");
    expect(result.messages).toEqual([]);
  });

  it("sets null activeDatasetId", () => {
    const state = {
      messages: [],
      inputHistory: [],
      historyCursor: null,
      isSending: false,
      modelOptions: [],
      selectedModelId: null,
      isModelsLoading: false,
      screenFeedback: null,
    };
    const result = mapChatStateWithDataset({ state, activeDatasetId: null });
    expect(result.activeDatasetId).toBeNull();
  });
});

describe("createOnMessageReceived", () => {
  it("calls addChartSpec for single chart spec", () => {
    const addChartSpec = vi.fn();
    const handler = createOnMessageReceived({ addChartSpec });
    const spec = { id: "c1", title: "T", type: "bar" as const, xKey: "x", yKeys: ["y"], data: [] };
    handler({ message: "done", chartSpec: spec });
    expect(addChartSpec).toHaveBeenCalledWith(spec);
  });

  it("calls addChartSpec for each spec in array in order (AC-03)", () => {
    const addChartSpec = vi.fn();
    const handler = createOnMessageReceived({ addChartSpec });
    const specs = [
      { id: "a", title: "A", type: "line" as const, xKey: "x", yKeys: ["y"], data: [] },
      { id: "b", title: "B", type: "bar" as const, xKey: "x", yKeys: ["y"], data: [] },
      { id: "c", title: "C", type: "area" as const, xKey: "x", yKeys: ["y"], data: [] },
    ];
    handler({ message: "charts", chartSpec: specs });
    expect(addChartSpec).toHaveBeenCalledTimes(3);
    expect(addChartSpec.mock.calls[0][0].id).toBe("a");
    expect(addChartSpec.mock.calls[1][0].id).toBe("b");
    expect(addChartSpec.mock.calls[2][0].id).toBe("c");
  });

  it("does not call addChartSpec when chartSpec is null", () => {
    const addChartSpec = vi.fn();
    const handler = createOnMessageReceived({ addChartSpec });
    handler({ message: "no chart", chartSpec: null });
    expect(addChartSpec).not.toHaveBeenCalled();
  });
});

describe("composeChatStateActions", () => {
  it("composes full ChatStateActions from core + chart", () => {
    const coreActions = {
      addMessage: vi.fn(),
      addInputToHistory: vi.fn(),
      moveHistoryCursor: vi.fn(),
      resetHistoryCursor: vi.fn(),
      setSending: vi.fn(),
      setModelOptions: vi.fn(),
      setSelectedModelId: vi.fn(),
      setModelsLoading: vi.fn(),
      setScreenFeedback: vi.fn(),
    } as unknown as ChatCoreStateActions;
    const addChartSpec = vi.fn();

    const result = composeChatStateActions({ coreActions, addChartSpec });
    expect(result.addChartSpec).toBe(addChartSpec);
    expect(result.onMessageReceived).toBeTypeOf("function");
    expect(result.addMessage).toBe(coreActions.addMessage);
  });
});
