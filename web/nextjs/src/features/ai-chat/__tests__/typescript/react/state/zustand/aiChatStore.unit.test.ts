import { beforeEach, describe, expect, it } from "vitest";
import { useAiChatStore } from "@/features/ai-chat/typescript/react/state/zustand/aiChatStore";
import { createInitialChatStoreCoreState } from "@/features/ai-chat/typescript/logic/chatStore.logic";

describe("aiChatStore", () => {
  beforeEach(() => {
    useAiChatStore.setState(createInitialChatStoreCoreState({}));
  });

  it("appends messages and input history", () => {
    useAiChatStore.getState().addMessage({
      id: "u1",
      role: "user",
      content: "hello",
      createdAt: 1,
    });
    useAiChatStore.getState().addInputToHistory("hello");

    expect(useAiChatStore.getState().messages).toHaveLength(1);
    expect(useAiChatStore.getState().inputHistory).toEqual(["hello"]);
    expect(useAiChatStore.getState().historyCursor).toBeNull();
  });

  it("navigates history cursor and resets cursor", () => {
    useAiChatStore.setState({ inputHistory: ["first", "second"], historyCursor: null });

    const up = useAiChatStore.getState().moveHistoryCursor("up");
    expect(up).toBe("second");
    expect(useAiChatStore.getState().historyCursor).toBe(1);

    const down = useAiChatStore.getState().moveHistoryCursor("down");
    expect(down).toBe("");
    expect(useAiChatStore.getState().historyCursor).toBeNull();

    useAiChatStore.setState({ historyCursor: 0 });
    useAiChatStore.getState().resetHistoryCursor();
    expect(useAiChatStore.getState().historyCursor).toBeNull();
  });

  it("updates sending/models slices", () => {
    useAiChatStore.getState().setSending(true);
    useAiChatStore.getState().setModelsLoading(true);
    useAiChatStore.getState().setModelOptions([{ id: "m1", label: "Model 1" }]);
    useAiChatStore.getState().setSelectedModelId("m1");

    expect(useAiChatStore.getState().isSending).toBe(true);
    expect(useAiChatStore.getState().isModelsLoading).toBe(true);
    expect(useAiChatStore.getState().modelOptions).toEqual([{ id: "m1", label: "Model 1" }]);
    expect(useAiChatStore.getState().selectedModelId).toBe("m1");
  });
});
