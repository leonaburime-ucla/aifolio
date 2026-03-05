import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const state = {
  messages: [],
  inputHistory: [],
  historyCursor: null,
  isSending: false,
  modelOptions: [],
  selectedModelId: null,
  isModelsLoading: false,
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
};

vi.mock("zustand/react/shallow", () => ({
  useShallow: <T,>(selector: T) => selector,
}));

vi.mock("@/features/ai-chat/typescript/react/state/zustand/aiChatStore", () => ({
  useAiChatStore: (selector: (store: typeof state & typeof actions) => unknown) =>
    selector({ ...state, ...actions }),
}));

import { useAiChatStateAdapter } from "@/features/ai-chat/typescript/react/state/adapters/aiChatState.adapter";

describe("aiChatState.adapter", () => {
  it("maps zustand store into chat state port", () => {
    const { result } = renderHook(() => useAiChatStateAdapter());

    expect(result.current.state).toEqual(state);
    expect(result.current.actions).toEqual(actions);
  });
});
