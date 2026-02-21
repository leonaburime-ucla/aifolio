import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { ChatAttachment, ChatMessage } from "@/features/ai/types/chat.types";

export type ChatSidebarUiDeps = {
  messages: ChatMessage[];
  isSending: boolean;
  addAttachments: (files: ChatAttachment[]) => void;
};

export type ChatSidebarUi = {
  scrollRef: RefObject<HTMLDivElement>;
  isDragging: boolean;
  copiedId: string | null;
  handleCopy: (id: string, content: string) => Promise<void>;
  handleDrop: (event: React.DragEvent<HTMLDivElement>) => Promise<void>;
  handleDragOver: (event: React.DragEvent<HTMLDivElement>) => void;
  handleDragLeave: () => void;
};

/**
 * UI-only behaviors for ChatSidebar (scrolling, drag/drop, copy feedback).
 * Keep this separate for easier unit testing.
 */
export function useChatSidebarUi(deps: ChatSidebarUiDeps): ChatSidebarUi {
  const { messages, isSending, addAttachments } = deps;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const raf = window.requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
    return () => window.cancelAnimationFrame(raf);
  }, [messages.length, isSending]);

  useEffect(() => {
    if (!copiedId) return;
    const timeout = window.setTimeout(() => setCopiedId(null), 2000);
    return () => window.clearTimeout(timeout);
  }, [copiedId]);

  const handleCopy = async (id: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(id);
    } catch (error) {
      setCopiedId(null);
    }
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length === 0) return;

    const attachments = await Promise.all(
      files.map(
        (file) =>
          new Promise<ChatAttachment>((resolve, reject) => {
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
          })
      )
    );

    addAttachments(attachments);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

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
