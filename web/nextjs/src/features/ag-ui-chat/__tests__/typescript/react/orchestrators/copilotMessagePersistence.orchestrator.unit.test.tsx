import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

let liveMessages: unknown[] = [];
const setMessages = vi.fn();

let persistedMessages: unknown[] = [];
let hasHydrated = true;
const setPersistedMessages = vi.fn();

vi.mock("@copilotkit/react-core", () => ({
  useCopilotChatInternal: () => ({
    messages: liveMessages,
    setMessages,
  }),
}));

vi.mock("@/features/ag-ui-chat/typescript/react/state/adapters/copilotMessageState.adapter", () => ({
  useCopilotMessageStateAdapter: () => ({
    messages: persistedMessages,
    hasHydrated,
    setMessages: setPersistedMessages,
  }),
}));

import { useCopilotMessagePersistenceOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/copilotMessagePersistence.orchestrator";

describe("useCopilotMessagePersistenceOrchestrator", () => {
  beforeEach(() => {
    liveMessages = [];
    persistedMessages = [];
    hasHydrated = true;
    setMessages.mockReset();
    setPersistedMessages.mockReset();
  });

  it("hydrates from persisted messages when live runtime is empty", () => {
    persistedMessages = [
      { id: "u1", type: "TextMessage", role: "user", content: "hello" },
      { id: "a1", type: "TextMessage", role: "assistant", content: "world" },
    ];

    renderHook(() => useCopilotMessagePersistenceOrchestrator());

    expect(setMessages).toHaveBeenCalledWith(persistedMessages as never[]);
  });

  it("retries restore while hydration is in progress and live messages are non-persistable", () => {
    persistedMessages = [
      { id: "u1", type: "TextMessage", role: "user", content: "hello" },
      { id: "a1", type: "TextMessage", role: "assistant", content: "world" },
    ];

    const { rerender } = renderHook(() => useCopilotMessagePersistenceOrchestrator());

    // Copilot transient runtime state after initial hydrate attempt.
    liveMessages = [
      { id: "coagent-state-render-agentic-research", role: "assistant", content: "" },
    ];
    rerender();

    expect(setMessages.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(setMessages).toHaveBeenLastCalledWith(persistedMessages as never[]);
  });

  it("does not hydrate when live runtime already has user-authored history", () => {
    liveMessages = [
      { id: "u1", type: "TextMessage", role: "user", content: "existing" },
    ];
    persistedMessages = [
      { id: "u2", type: "TextMessage", role: "user", content: "persisted" },
    ];

    renderHook(() => useCopilotMessagePersistenceOrchestrator());

    expect(setMessages).not.toHaveBeenCalled();
  });
});
