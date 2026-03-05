import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useChatSidebarUi } from "@/features/ai-chat/typescript/react/hooks/useChatSidebar.web";

function createDropEvent(files: File[]): React.DragEvent<HTMLDivElement> {
  return {
    preventDefault: vi.fn(),
    dataTransfer: { files } as unknown as DataTransfer,
  } as React.DragEvent<HTMLDivElement>;
}

describe("REQ-008/ERR-007 invalid attachment handling", () => {
  it("does not throw and does not add attachments when file read fails", async () => {
    const addAttachments = vi.fn();

    const originalFileReader = globalThis.FileReader;
    class FailingFileReader {
      public result: string | ArrayBuffer | null = null;
      public error: DOMException | null = new DOMException("read failed");
      public onload: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;
      public onerror: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;

      public readAsDataURL(): void {
        this.onerror?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>);
      }
    }

    // Force deterministic invalid-attachment scenario.
    globalThis.FileReader = FailingFileReader as unknown as typeof FileReader;
    try {
      const { result } = renderHook(() =>
        useChatSidebarUi({
          messages: [],
          isSending: false,
          addAttachments,
        })
      );

      const event = createDropEvent([
        new File(["bad"], "bad.txt", { type: "text/plain" }),
      ]);

      await expect(result.current.handleDrop(event)).resolves.toBeUndefined();
      expect(addAttachments).not.toHaveBeenCalled();
    } finally {
      globalThis.FileReader = originalFileReader;
    }
  });
});
