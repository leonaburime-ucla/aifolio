import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatSidebarUi } from "@/features/ai-chat/typescript/react/hooks/useChatSidebar.web";
import type { ChatAttachment } from "@/features/ai-chat/__types__/typescript/chat.types";

function createDropEvent(files: File[]): React.DragEvent<HTMLDivElement> {
  return {
    preventDefault: vi.fn(),
    dataTransfer: { files } as unknown as DataTransfer,
  } as React.DragEvent<HTMLDivElement>;
}

describe("useChatSidebarUi runtime deps", () => {
  it("uses injected clipboard writer", async () => {
    const writeClipboard = vi.fn(async () => undefined);
    const { result } = renderHook(() =>
      useChatSidebarUi(
        {
          messages: [],
          isSending: false,
          addAttachments: vi.fn(),
        },
        { writeClipboard }
      )
    );

    await act(async () => {
      await result.current.handleCopy("m1", "copy me");
    });

    expect(writeClipboard).toHaveBeenCalledWith("copy me");
    expect(result.current.copiedId).toBe("m1");
  });

  it("uses injected file reader adapter and only forwards fulfilled attachments", async () => {
    const addAttachments = vi.fn();
    const readFileAsDataUrl = vi.fn(
      async (file: File): Promise<ChatAttachment> => {
        if (file.name === "bad.txt") {
          throw new Error("read failed");
        }
        return {
          name: file.name,
          type: file.type || "application/octet-stream",
          size: file.size,
          dataUrl: "data:text/plain;base64,b2s=",
        };
      }
    );

    const { result } = renderHook(() =>
      useChatSidebarUi(
        {
          messages: [],
          isSending: false,
          addAttachments,
        },
        { readFileAsDataUrl }
      )
    );

    const event = createDropEvent([
      new File(["ok"], "ok.txt", { type: "text/plain" }),
      new File(["bad"], "bad.txt", { type: "text/plain" }),
    ]);

    await act(async () => {
      await result.current.handleDrop(event);
    });

    expect(readFileAsDataUrl).toHaveBeenCalledTimes(2);
    expect(addAttachments).toHaveBeenCalledTimes(1);
    expect(addAttachments).toHaveBeenCalledWith([
      expect.objectContaining({ name: "ok.txt" }),
    ]);
  });

  it("toggles dragging state via drag handlers", () => {
    const { result } = renderHook(() =>
      useChatSidebarUi({
        messages: [],
        isSending: false,
        addAttachments: vi.fn(),
      })
    );

    const dragEvent = {
      preventDefault: vi.fn(),
    } as unknown as React.DragEvent<HTMLDivElement>;

    act(() => {
      result.current.handleDragOver(dragEvent);
    });
    expect(dragEvent.preventDefault).toHaveBeenCalledTimes(1);
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.handleDragLeave();
    });
    expect(result.current.isDragging).toBe(false);
  });

  it("uses default FileReader mapper and RAF cleanup path", async () => {
    const addAttachments = vi.fn();
    const originalFileReader = globalThis.FileReader;
    class SuccessfulFileReader {
      public result: string | ArrayBuffer | null = "data:text/plain;base64,b2s=";
      public error: DOMException | null = null;
      public onload: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;
      public onerror: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;

      public readAsDataURL(): void {
        this.onload?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>);
      }
    }
    globalThis.FileReader = SuccessfulFileReader as unknown as typeof FileReader;

    const requestAnimationFrameImpl = vi.fn((cb: FrameRequestCallback) => {
      cb(0);
      return 1;
    });
    const cancelAnimationFrameImpl = vi.fn();

    try {
      const { result, rerender, unmount } = renderHook(
        ({ messages }) =>
          useChatSidebarUi(
            {
              messages,
              isSending: false,
              addAttachments,
            },
            { requestAnimationFrameImpl, cancelAnimationFrameImpl }
          ),
        {
          initialProps: {
            messages: [] as Array<{
              id: string;
              role: "user" | "assistant";
              content: string;
              createdAt: number;
            }>,
          },
        }
      );

      result.current.scrollRef.current = {
        scrollTop: 0,
        scrollHeight: 100,
      } as unknown as HTMLDivElement;

      rerender({
        messages: [{ id: "m1", role: "user", content: "hi", createdAt: 1 }],
      });

      const event = createDropEvent([
        new File(["ok"], "ok.bin"),
      ]);

      await act(async () => {
        await result.current.handleDrop(event);
      });

      expect(requestAnimationFrameImpl).toHaveBeenCalled();
      expect(addAttachments).toHaveBeenCalledWith([
        expect.objectContaining({
          name: "ok.bin",
          type: "application/octet-stream",
          dataUrl: "data:text/plain;base64,b2s=",
        }),
      ]);

      unmount();
      expect(cancelAnimationFrameImpl).toHaveBeenCalled();
    } finally {
      globalThis.FileReader = originalFileReader;
    }
  });

  it("returns early when dropped file list is empty", async () => {
    const addAttachments = vi.fn();
    const { result } = renderHook(() =>
      useChatSidebarUi({
        messages: [],
        isSending: false,
        addAttachments,
      })
    );

    await act(async () => {
      await result.current.handleDrop(createDropEvent([]));
    });

    expect(addAttachments).not.toHaveBeenCalled();
  });

  it("returns early when dataTransfer.files is missing", async () => {
    const addAttachments = vi.fn();
    const { result } = renderHook(() =>
      useChatSidebarUi({
        messages: [],
        isSending: false,
        addAttachments,
      })
    );

    const event = {
      preventDefault: vi.fn(),
      dataTransfer: {} as DataTransfer,
    } as React.DragEvent<HTMLDivElement>;

    await act(async () => {
      await result.current.handleDrop(event);
    });

    expect(addAttachments).not.toHaveBeenCalled();
  });
});
