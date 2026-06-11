import { describe, it, expect } from "vitest";
import {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
} from "../../../src/chat/index";

describe("normalizeSubmissionValue", () => {
  it("trims whitespace and returns value", () => {
    expect(normalizeSubmissionValue({ value: "  hello  " })).toBe("hello");
  });

  it("returns null for empty string", () => {
    expect(normalizeSubmissionValue({ value: "" })).toBeNull();
  });

  it("returns null for whitespace-only string", () => {
    expect(normalizeSubmissionValue({ value: "   " })).toBeNull();
  });

  it("returns null for tab/newline only", () => {
    expect(normalizeSubmissionValue({ value: "\t\n" })).toBeNull();
  });
});

describe("buildChatHistoryWindow", () => {
  const messages = Array.from({ length: 12 }, (_, i) => ({
    id: `msg-${i}`,
    role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
    content: `message ${i}`,
    createdAt: 1000 + i,
  }));

  it("returns at most 10 messages by default (including current)", () => {
    const result = buildChatHistoryWindow({
      messages,
      userContent: "new message",
      attachments: undefined,
    });
    expect(result.length).toBe(10);
    expect(result[result.length - 1].content).toBe("new message");
  });

  it("respects custom windowSize", () => {
    const result = buildChatHistoryWindow(
      { messages, userContent: "hi", attachments: undefined },
      { windowSize: 3 }
    );
    expect(result.length).toBe(3);
  });

  it("includes current user message as last entry", () => {
    const result = buildChatHistoryWindow({
      messages: [],
      userContent: "test",
      attachments: undefined,
    });
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ role: "user", content: "test", attachments: undefined });
  });

  it("passes attachments on current message", () => {
    const attachments = [{ name: "f.txt", type: "text/plain", size: 5, dataUrl: "data:" }];
    const result = buildChatHistoryWindow({
      messages: [],
      userContent: "with file",
      attachments,
    });
    expect(result[0].attachments).toBe(attachments);
  });
});

describe("createUserChatMessage", () => {
  it("creates a user message", () => {
    const msg = createUserChatMessage({
      id: "u1",
      content: "hello",
      createdAt: 1000,
    });
    expect(msg.role).toBe("user");
    expect(msg.id).toBe("u1");
    expect(msg.content).toBe("hello");
    expect(msg.createdAt).toBe(1000);
  });
});

describe("createAssistantChatMessage", () => {
  it("creates an assistant message with null chartSpec", () => {
    const msg = createAssistantChatMessage({
      id: "a1",
      content: "response",
      createdAt: 2000,
    });
    expect(msg.role).toBe("assistant");
    expect(msg.chartSpec).toBeNull();
  });
});

describe("shouldRestoreDraftValue", () => {
  it("returns true when direction is down, cursor is set, and next value is empty", () => {
    expect(
      shouldRestoreDraftValue({
        direction: "down",
        historyCursor: 1,
        nextValue: "",
      })
    ).toBe(true);
  });

  it("returns false when direction is up", () => {
    expect(
      shouldRestoreDraftValue({
        direction: "up",
        historyCursor: 1,
        nextValue: "",
      })
    ).toBe(false);
  });

  it("returns false when cursor is null", () => {
    expect(
      shouldRestoreDraftValue({
        direction: "down",
        historyCursor: null,
        nextValue: "",
      })
    ).toBe(false);
  });

  it("returns false when next value is non-empty", () => {
    expect(
      shouldRestoreDraftValue({
        direction: "down",
        historyCursor: 1,
        nextValue: "something",
      })
    ).toBe(false);
  });
});
