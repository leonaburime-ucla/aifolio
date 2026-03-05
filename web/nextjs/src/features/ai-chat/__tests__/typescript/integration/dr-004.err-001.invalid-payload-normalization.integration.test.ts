import { describe, expect, it, vi } from "vitest";
import { sendChatMessageDirect } from "@/features/ai-chat/typescript/api/chatApi";

describe("DR-004 / ERR-001 invalid payload normalization", () => {
  it("returns null for invalid assistant payload shapes without throwing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "ok",
        result: { message: 42, chartSpec: null },
      }),
    } as Response);

    await expect(
      sendChatMessageDirect({
        value: "hello",
        model: null,
        history: [],
        attachments: [],
      })
    ).resolves.toBeNull();
  });

  it("returns null when gemini parts payload has no text", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "ok",
        result: [{ type: "text" }],
      }),
    } as Response);

    await expect(
      sendChatMessageDirect({
        value: "hello",
        model: null,
        history: [],
        attachments: [],
      })
    ).resolves.toBeNull();
  });
});
