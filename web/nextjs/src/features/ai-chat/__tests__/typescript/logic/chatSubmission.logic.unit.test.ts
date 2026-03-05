import { describe, expect, it } from "vitest";
import {
  buildChatHistoryWindow,
  createAssistantChatMessage,
  createUserChatMessage,
  normalizeSubmissionValue,
  shouldRestoreDraftValue,
} from "@/features/ai-chat/typescript/logic/chatSubmission.logic";

describe("chatSubmission.logic", () => {
  it("normalizes empty and non-empty submission values", () => {
    expect(normalizeSubmissionValue({ value: "  hello  " })).toBe("hello");
    expect(normalizeSubmissionValue({ value: "   " })).toBeNull();
  });

  it("builds bounded history with current user message at tail", () => {
    const history = buildChatHistoryWindow(
      {
        messages: [
          { id: "1", role: "user", content: "a", createdAt: 1 },
          { id: "2", role: "assistant", content: "b", createdAt: 2 },
        ],
        userContent: "c",
        attachments: [
          { name: "n", type: "text/plain", size: 1, dataUrl: "data:text/plain,a" },
        ],
      },
      { windowSize: 2 }
    );

    expect(history).toEqual([
      { role: "assistant", content: "b" },
      {
        role: "user",
        content: "c",
        attachments: [
          { name: "n", type: "text/plain", size: 1, dataUrl: "data:text/plain,a" },
        ],
      },
    ]);
  });

  it("creates user and assistant messages with expected defaults", () => {
    expect(
      createUserChatMessage({ id: "u1", content: "hi", createdAt: 10 })
    ).toEqual({
      id: "u1",
      role: "user",
      content: "hi",
      createdAt: 10,
    });

    expect(
      createAssistantChatMessage({ id: "a1", content: "hello", createdAt: 11 })
    ).toEqual({
      id: "a1",
      role: "assistant",
      content: "hello",
      chartSpec: null,
      createdAt: 11,
    });
  });

  it("restores draft only for down-navigation off a selected history value", () => {
    expect(
      shouldRestoreDraftValue({
        direction: "down",
        historyCursor: 1,
        nextValue: "",
      })
    ).toBe(true);

    expect(
      shouldRestoreDraftValue({
        direction: "up",
        historyCursor: 1,
        nextValue: "",
      })
    ).toBe(false);
  });
});
