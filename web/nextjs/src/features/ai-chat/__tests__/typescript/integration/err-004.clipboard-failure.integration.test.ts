import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatSidebarUi } from "@/features/ai-chat/typescript/react/hooks/useChatSidebar.web";

describe("ERR-004 clipboard failure", () => {
  it("does not throw and keeps copied indicator cleared when clipboard write fails", async () => {
    const writeText = vi.fn(async () => {
      throw new Error("denied");
    });

    Object.defineProperty(globalThis.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    const { result } = renderHook(() =>
      useChatSidebarUi({
        messages: [],
        isSending: false,
        addAttachments: vi.fn(),
      })
    );

    await act(async () => {
      await expect(result.current.handleCopy("m1", "hello")).resolves.toBeUndefined();
    });

    expect(writeText).toHaveBeenCalledWith("hello");
    expect(result.current.copiedId).toBeNull();
  });
});
