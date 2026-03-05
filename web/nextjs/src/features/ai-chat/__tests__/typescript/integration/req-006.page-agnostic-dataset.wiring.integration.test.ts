import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import type { ChatDeps, ChatIntegration } from "@/features/ai-chat/__types__/typescript/chat.types";

const { useChatIntegrationMock } = vi.hoisted(() => ({
  useChatIntegrationMock: vi.fn(
    (deps: ChatDeps): ChatIntegration => deps as unknown as ChatIntegration
  ),
}));

const stateAdapterMock = {
  state: {
    messages: [],
    inputHistory: [],
    historyCursor: null,
    isSending: false,
    modelOptions: [],
    selectedModelId: null,
    isModelsLoading: false,
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
  },
};

vi.mock("@/features/ai-chat/typescript/react/hooks/useChat.hooks", () => ({
  useChatIntegration: useChatIntegrationMock,
}));

vi.mock("@/features/ai-chat/typescript/react/state/adapters/aiChatState.adapter", () => ({
  useAiChatStateAdapter: () => stateAdapterMock,
}));

vi.mock("@/features/ai-chat/typescript/api/chatApi.adapter", () => ({
  createChatApiAdapter: () => ({
    sendMessage: vi.fn(async () => null),
    fetchModels: vi.fn(async () => null),
  }),
}));

import { useChatOrchestrator } from "@/features/ai-chat/typescript/react/orchestrators/chatOrchestrator";

describe("REQ-006 chat orchestrator page-agnostic dataset wiring", () => {
  it("injects activeDatasetId as null by default", () => {
    const { result } = renderHook(() => useChatOrchestrator());

    const deps = useChatIntegrationMock.mock.calls[0]?.[0] as ChatDeps;

    expect(deps).toBeDefined();
    expect(deps.state.activeDatasetId).toBeNull();
    expect(result.current).toBeDefined();
  });
});
