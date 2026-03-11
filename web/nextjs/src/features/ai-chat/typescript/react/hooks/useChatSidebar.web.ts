import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { ChatAttachment, ChatMessage } from "@/features/ai-chat/__types__/typescript/chat.types";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
}

export type ChatSidebarUiDeps = {
  messages: ChatMessage[];
  isSending: boolean;
  addAttachments: (files: ChatAttachment[]) => void;
};

export type ChatSidebarUi = {
  scrollRef: RefObject<HTMLDivElement | null>;
  isDragging: boolean;
  copiedId: string | null;
  handleCopy: (id: string, content: string) => Promise<void>;
  handleDrop: (event: React.DragEvent<HTMLDivElement>) => Promise<void>;
  handleDragOver: (event: React.DragEvent<HTMLDivElement>) => void;
  handleDragLeave: () => void;
};

export type ChatSidebarRuntimeDeps = {
  requestAnimationFrameImpl?: typeof window.requestAnimationFrame;
  cancelAnimationFrameImpl?: typeof window.cancelAnimationFrame;
  setTimeoutImpl?: typeof window.setTimeout;
  clearTimeoutImpl?: typeof window.clearTimeout;
  writeClipboard?: (content: string) => Promise<void>;
  readFileAsDataUrl?: (file: File) => Promise<ChatAttachment>;
};

function defaultReadFileAsDataUrl(file: File): Promise<ChatAttachment> {
  return new Promise<ChatAttachment>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve({
        name: file.name,
        type: file.type || "application/octet-stream",
        size: file.size,
        dataUrl: String(reader.result),
      });
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/**
 * UI-only behaviors for ChatSidebar (scrolling, drag/drop, copy feedback).
 * Keep this separate for easier unit testing and view reuse.
 *
 * @param deps - Required sidebar dependencies from chat integration state/actions.
 * @returns Sidebar UI state and event handlers.
 */
export function useChatSidebarUi(
  deps: ChatSidebarUiDeps,
  runtime?: ChatSidebarRuntimeDeps
): ChatSidebarUi {
  const { messages, isSending, addAttachments } = deps;
  const requestAnimationFrameImpl =
    runtime?.requestAnimationFrameImpl ?? window.requestAnimationFrame;
  const cancelAnimationFrameImpl =
    runtime?.cancelAnimationFrameImpl ?? window.cancelAnimationFrame;
  const setTimeoutImpl = runtime?.setTimeoutImpl ?? window.setTimeout;
  const clearTimeoutImpl = runtime?.clearTimeoutImpl ?? window.clearTimeout;
  const writeClipboard =
    runtime?.writeClipboard ??
    ((content: string) => navigator.clipboard.writeText(content));
  const readFileAsDataUrl = runtime?.readFileAsDataUrl ?? defaultReadFileAsDataUrl;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  /**
   * Auto-scroll to latest message whenever message count changes or
   * sending state toggles.
   */
  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[chat-debug] sidebar_autoscroll_effect", {
        path: getDebugPath(),
        messageCount: messages.length,
        isSending,
      });
    }
    const container = scrollRef.current;
    if (!container) return;
    const raf = requestAnimationFrameImpl(() => {
      container.scrollTop = container.scrollHeight;
    });
    return () => cancelAnimationFrameImpl(raf);
  }, [messages.length, isSending, requestAnimationFrameImpl, cancelAnimationFrameImpl]);

  /** Reset transient "copied" badge state after 2 seconds. */
  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[chat-debug] sidebar_copied_badge_effect", {
        path: getDebugPath(),
        copiedId,
      });
    }
    if (!copiedId) return;
    const timeout = setTimeoutImpl(() => setCopiedId(null), 2000);
    return () => clearTimeoutImpl(timeout);
  }, [copiedId, setTimeoutImpl, clearTimeoutImpl]);

  /**
   * Copy assistant/user content to clipboard and expose temporary copied state.
   *
   * @param id - Required message identifier for copied state tracking.
   * @param content - Required message content to copy.
   * @returns Promise that resolves when copy attempt completes.
   */
  const handleCopy = async (id: string, content: string): Promise<void> => {
    try {
      await writeClipboard(content);
      setCopiedId(id);
    } catch (error) {
      setCopiedId(null);
    }
  };

  /**
   * Convert dropped files into attachment records and stage them for submit.
   *
   * @param event - Required drop event carrying attached files.
   * @returns Promise that resolves after files are transformed and staged.
   */
  const handleDrop = async (
    event: React.DragEvent<HTMLDivElement>
  ): Promise<void> => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length === 0) return;

    const results = await Promise.allSettled(
      files.map((file) => readFileAsDataUrl(file))
    );

    const attachments = results
      .filter(
        (result): result is PromiseFulfilledResult<ChatAttachment> =>
          result.status === "fulfilled"
      )
      .map((result) => result.value);

    if (attachments.length === 0) return;
    addAttachments(attachments);
  };

  /**
   * Marks drop-zone state while drag is active.
   *
   * @param event - Required drag-over event.
   * @returns void
   */
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setIsDragging(true);
  };

  /**
   * Clears drag-over visual state when pointer leaves drop-zone.
   *
   * @returns void
   */
  const handleDragLeave = (): void => setIsDragging(false);

  return {
    scrollRef,
    isDragging,
    copiedId,
    handleCopy,
    handleDrop,
    handleDragOver,
    handleDragLeave,
  };
}
