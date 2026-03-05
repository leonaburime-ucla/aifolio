import { describe, expect, it } from "vitest";
import {
  appendInputHistory,
  appendMessage,
  createInitialChatStoreCoreState,
  resolveHistoryCursor,
} from "@/features/ai-chat/typescript/logic/chatStore.logic";
import type { ChatMessage } from "@/features/ai-chat/__types__/typescript/chat.types";

describe("chatStore.logic", () => {
  it("creates default state shape", () => {
    expect(createInitialChatStoreCoreState({})).toEqual({
      messages: [],
      inputHistory: [],
      historyCursor: null,
      isSending: false,
      modelOptions: [],
      selectedModelId: null,
      isModelsLoading: false,
    });
  });

  it("appends messages and input history deterministically", () => {
    const message = {
      id: "m1",
      role: "user",
      content: "hi",
      createdAt: 1,
    } satisfies ChatMessage;

    expect(appendMessage({ messages: [], message })).toEqual([message]);
    expect(
      appendInputHistory({ inputHistory: ["a"], value: "b" })
    ).toEqual({
      inputHistory: ["a", "b"],
      historyCursor: null,
    });
  });

  it("resolves history cursor up/down across boundaries", () => {
    const history = ["one", "two", "three"];

    expect(
      resolveHistoryCursor({
        inputHistory: history,
        historyCursor: null,
        direction: "up",
      })
    ).toEqual({ nextCursor: 2, value: "three" });

    expect(
      resolveHistoryCursor({
        inputHistory: history,
        historyCursor: 2,
        direction: "up",
      })
    ).toEqual({ nextCursor: 1, value: "two" });

    expect(
      resolveHistoryCursor({
        inputHistory: history,
        historyCursor: 1,
        direction: "down",
      })
    ).toEqual({ nextCursor: 2, value: "three" });

    expect(
      resolveHistoryCursor({
        inputHistory: history,
        historyCursor: 2,
        direction: "down",
      })
    ).toEqual({ nextCursor: null, value: "" });
  });

  it("returns safe empty values for empty history", () => {
    expect(
      resolveHistoryCursor({
        inputHistory: [],
        historyCursor: 7,
        direction: "up",
      })
    ).toEqual({ nextCursor: 7, value: "" });
  });

  it("returns empty value for sparse history holes in up/down branches", () => {
    const sparse = ["one", , "three"] as unknown as string[];

    expect(
      resolveHistoryCursor({
        inputHistory: sparse,
        historyCursor: 2,
        direction: "up",
      })
    ).toEqual({ nextCursor: 1, value: "" });

    expect(
      resolveHistoryCursor({
        inputHistory: sparse,
        historyCursor: 0,
        direction: "down",
      })
    ).toEqual({ nextCursor: 1, value: "" });
  });
});
