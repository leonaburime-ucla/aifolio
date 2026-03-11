import { describe, expect, it, vi } from "vitest";
import {
  createChatApiDeps,
  createChatDeps,
} from "@/features/ai-chat/typescript/logic/chatOrchestrator.logic";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

describe("chatOrchestrator.logic", () => {
  it("creates API deps without mutation", () => {
    const sendMessage = vi.fn(async () => null);
    const fetchModels = vi.fn(async () => null);

    const api = createChatApiDeps({ sendMessage, fetchModels });

    expect(api.sendMessage).toBe(sendMessage);
    expect(api.fetchModels).toBe(fetchModels);
  });

  it("creates chat deps bundle with stable references", () => {
    const state = {
      messages: [],
      inputHistory: [],
      historyCursor: null,
      isSending: false,
      modelOptions: [],
      selectedModelId: null,
      isModelsLoading: false,
      activeDatasetId: null,
    };
    const actions = {
      addMessage: vi.fn(),
      addInputToHistory: vi.fn(),
      moveHistoryCursor: vi.fn(() => ""),
      resetHistoryCursor: vi.fn(),
      setSending: vi.fn(),
      setModelOptions: vi.fn(),
      setSelectedModelId: vi.fn(),
      setModelsLoading: vi.fn(),
      addChartSpec: vi.fn(),
      onMessageReceived: vi.fn(),
    };
    const api = {
      sendMessage: vi.fn(async () => null),
      fetchModels: vi.fn(async () => null),
    };

    const deps = createChatDeps({ state, actions, api, logic: DEFAULT_CHAT_LOGIC_DEPS });

    expect(deps.state).toBe(state);
    expect(deps.actions).toBe(actions);
    expect(deps.api).toBe(api);
    expect(deps.logic).toBe(DEFAULT_CHAT_LOGIC_DEPS);
  });
});
