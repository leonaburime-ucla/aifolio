import { describe, expect, it } from "vitest";
import { buildChatHistoryWindow } from "@/features/ai-chat/typescript/logic/chatSubmission.logic";
import type { ChatMessage } from "@/features/ai-chat/__types__/typescript/chat.types";

function createMessages(count: number): ChatMessage[] {
  return Array.from({ length: count }, (_, index) => ({
    id: String(index),
    role: index % 2 === 0 ? "user" : "assistant",
    content: `m-${index}`,
    createdAt: index,
  }));
}

describe("DR-002 history payload window", () => {
  it("returns at most 10 entries and includes current user message", () => {
    const history = buildChatHistoryWindow({
      messages: createMessages(15),
      userContent: "latest-user",
      attachments: [{ name: "a.txt", type: "text/plain", size: 1, dataUrl: "x" }],
    });

    expect(history).toHaveLength(10);
    expect(history[history.length - 1]).toEqual({
      role: "user",
      content: "latest-user",
      attachments: [{ name: "a.txt", type: "text/plain", size: 1, dataUrl: "x" }],
    });
  });
});
