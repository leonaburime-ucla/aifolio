import { describe, it, expect } from "vitest";
import {
  ChatMessageSchema,
  ChatModelOptionSchema,
  ChatAssistantPayloadSchema,
  ChatAttachmentSchema,
  ChatHistoryMessageSchema,
  ScreenFeedbackSchema,
} from "../src/entities/chat/index.ts";

describe("ChatMessageSchema", () => {
  const validMessage = {
    id: "msg-1",
    role: "user" as const,
    content: "Hello",
    createdAt: 1717900000000,
  };

  it("validates a user message", () => {
    const result = ChatMessageSchema.parse(validMessage);
    expect(result.role).toBe("user");
    expect(result.content).toBe("Hello");
  });

  it("validates an assistant message with chartSpec null", () => {
    const result = ChatMessageSchema.parse({
      ...validMessage,
      role: "assistant",
      chartSpec: null,
    });
    expect(result.chartSpec).toBeNull();
  });

  it("rejects invalid role", () => {
    expect(() =>
      ChatMessageSchema.parse({ ...validMessage, role: "system" })
    ).toThrow();
  });

  it("rejects missing id", () => {
    const { id: _, ...noId } = validMessage;
    expect(() => ChatMessageSchema.parse(noId)).toThrow();
  });

  it("rejects missing createdAt", () => {
    const { createdAt: _, ...noTimestamp } = validMessage;
    expect(() => ChatMessageSchema.parse(noTimestamp)).toThrow();
  });
});

describe("ChatModelOptionSchema", () => {
  it("validates a model option", () => {
    const result = ChatModelOptionSchema.parse({
      id: "gemini-3-flash",
      label: "Gemini 3 Flash",
    });
    expect(result.id).toBe("gemini-3-flash");
  });

  it("rejects missing label", () => {
    expect(() => ChatModelOptionSchema.parse({ id: "x" })).toThrow();
  });
});

describe("ChatAssistantPayloadSchema", () => {
  it("validates payload with null chartSpec", () => {
    const result = ChatAssistantPayloadSchema.parse({
      message: "Here's your answer",
      chartSpec: null,
    });
    expect(result.message).toBe("Here's your answer");
    expect(result.chartSpec).toBeNull();
  });

  it("validates payload with single chart spec", () => {
    const result = ChatAssistantPayloadSchema.parse({
      message: "Chart ready",
      chartSpec: {
        id: "c1",
        title: "Test",
        type: "bar",
        xKey: "x",
        yKeys: ["y"],
        data: [{ x: "a", y: 1 }],
      },
    });
    expect(result.chartSpec).toBeDefined();
  });

  it("validates payload with array of chart specs", () => {
    const result = ChatAssistantPayloadSchema.parse({
      message: "Multiple charts",
      chartSpec: [
        { id: "c1", title: "A", type: "line", xKey: "x", yKeys: ["y"], data: [] },
        { id: "c2", title: "B", type: "bar", xKey: "x", yKeys: ["y"], data: [] },
      ],
    });
    expect(Array.isArray(result.chartSpec)).toBe(true);
  });
});

describe("ChatAttachmentSchema", () => {
  it("validates an attachment", () => {
    const result = ChatAttachmentSchema.parse({
      name: "file.csv",
      type: "text/csv",
      size: 1024,
      dataUrl: "data:text/csv;base64,abc",
    });
    expect(result.name).toBe("file.csv");
  });
});

describe("ChatHistoryMessageSchema", () => {
  it("validates without attachments", () => {
    const result = ChatHistoryMessageSchema.parse({
      role: "user",
      content: "test",
    });
    expect(result.attachments).toBeUndefined();
  });

  it("validates with attachments", () => {
    const result = ChatHistoryMessageSchema.parse({
      role: "user",
      content: "test",
      attachments: [{ name: "f.txt", type: "text/plain", size: 10, dataUrl: "data:" }],
    });
    expect(result.attachments?.length).toBe(1);
  });
});

describe("ScreenFeedbackSchema", () => {
  it("validates minimal feedback", () => {
    const result = ScreenFeedbackSchema.parse({
      kind: "error",
      code: "MODEL_FETCH_TIMEOUT",
      message: "Request timed out",
    });
    expect(result.kind).toBe("error");
  });

  it("validates with optional fields", () => {
    const result = ScreenFeedbackSchema.parse({
      kind: "warning",
      code: "SLOW_RESPONSE",
      message: "Taking longer than usual",
      retryable: true,
      actionLabel: "Retry",
    });
    expect(result.retryable).toBe(true);
    expect(result.actionLabel).toBe("Retry");
  });

  it("rejects invalid kind", () => {
    expect(() =>
      ScreenFeedbackSchema.parse({ kind: "fatal", code: "x", message: "y" })
    ).toThrow();
  });
});
