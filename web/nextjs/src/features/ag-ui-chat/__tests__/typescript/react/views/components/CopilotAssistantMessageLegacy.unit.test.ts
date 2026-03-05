import { describe, expect, it } from "vitest";

import { toLegacyAssistantRenderMessage } from "@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util";

type LegacyMessageInput = Parameters<typeof toLegacyAssistantRenderMessage>[0];

describe("toLegacyAssistantRenderMessage", () => {
  it("returns transformed content for string messages", () => {
    const message = {
      id: "m1",
      role: "assistant",
      content: '{"message":"Hello"}',
    } as LegacyMessageInput;

    const result = toLegacyAssistantRenderMessage(message);

    expect(result).toEqual({
      ...message,
      content: "Hello",
    });
  });

  it("returns message unchanged when content is not a string", () => {
    const message = {
      id: "m2",
      role: "assistant",
      content: [{ type: "text", text: "Hi" }],
    } as LegacyMessageInput;

    const result = toLegacyAssistantRenderMessage(message);
    expect(result).toBe(message);
  });

  it("returns nullish message unchanged", () => {
    expect(toLegacyAssistantRenderMessage(undefined)).toBeUndefined();
    expect(toLegacyAssistantRenderMessage(null as LegacyMessageInput)).toBeNull();
  });
});
