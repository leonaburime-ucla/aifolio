import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import type { ChatApiDeps, ChatDeps, ChatStatePort } from "@aifolio/contracts/entities/chat";
import type { ChatIntegration } from "@aifolio/contracts/entities/chat";

const { useChatIntegrationMock } = vi.hoisted(() => ({
  useChatIntegrationMock: vi.fn(
    (deps: ChatDeps): ChatIntegration => deps as unknown as ChatIntegration
  ),
}));

const stateAdapterMock: ChatStatePort = {
  state: {
    messages: [],
    inputHistory: [],
    historyCursor: null,
    isSending: false,
    modelOptions: [],
    selectedModelId: null,
    isModelsLoading: false,
    screenFeedback: null,
  },
  actions: {
    addMessage: vi.fn(),
    addInputToHistory: vi.fn(),
    moveHistoryCursor: vi.fn(() => ""),
    resetHistoryCursor: vi.fn(),
    setSending: vi.fn(),
    setModelOptions: vi.fn(),
    setSelectedModelId: vi.fn(),
    setModelsLoading: vi.fn(),
    setScreenFeedback: vi.fn(),
  },
};

vi.mock("@/features/ai-chat/react/hooks/useChat.hooks", () => ({
  useChatIntegration: useChatIntegrationMock,
}));

vi.mock("@/features/ai-chat/react/state/adapters/aiChatState.adapter", () => ({
  useAiChatStateAdapter: () => stateAdapterMock,
}));

vi.mock("@/features/ai-chat/api/chatApi.adapter", () => ({
  createChatApiAdapter: () => ({
    sendMessage: vi.fn(async () => null),
    fetchModels: vi.fn(async () => null),
  }),
}));

import { useChatOrchestrator } from "@/features/ai-chat/react/orchestrators/chatOrchestrator";

describe("chatOrchestrator", () => {
  it("uses injected chart actions and api adapter when provided", () => {
    const addChartSpec = vi.fn();
    const apiAdapter: ChatApiDeps = {
      sendMessage: vi.fn(async () => null),
      fetchModels: vi.fn(async () => null),
    };

    renderHook(() =>
      useChatOrchestrator({
        chartActionsPort: { addChartSpec },
        apiAdapter,
      })
    );

    const deps = useChatIntegrationMock.mock.calls[0]?.[0] as ChatDeps;
    expect(deps.api.sendMessage).toBe(apiAdapter.sendMessage);
    expect(deps.api.fetchModels).toBe(apiAdapter.fetchModels);
    expect(deps.actions.addChartSpec).toBe(addChartSpec);

    deps.actions.onMessageReceived({
      message: "hello",
      chartSpec: {
        id: "chart-1",
        title: "A",
        type: "line",
        xKey: "x",
        yKeys: ["y"],
        data: [],
      },
    });
    expect(addChartSpec).toHaveBeenCalledTimes(1);
  });
});
