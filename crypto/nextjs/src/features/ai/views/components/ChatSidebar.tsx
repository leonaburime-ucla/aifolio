"use client";

import ChatBar from "@/features/ai/views/components/ChatBar";
import { useChatOrchestrator } from "@/features/ai/orchestrators/chatOrchestrator";
import { useChatSidebarUi } from "@/features/ai/hooks/useChatSidebar.web";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Sidebar chat panel modeled after VSCode AI chat.
 */
export default function ChatSidebar() {
  const {
    messages,
    isSending,
    modelOptions,
    selectedModelId,
    isModelsLoading,
    setSelectedModelId,
    addAttachments,
    attachments,
    removeAttachment,
  } = useChatOrchestrator();
  const {
    scrollRef,
    isDragging,
    copiedId,
    handleCopy,
    handleDrop,
    handleDragOver,
    handleDragLeave,
  } = useChatSidebarUi({ messages, isSending, addAttachments });

  return (
    <aside
      className="relative flex h-[calc(100vh-64px)] w-full flex-col border-l border-zinc-200 bg-white"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging ? (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-zinc-900/20 text-lg font-semibold text-white">
          Drop files to attach
        </div>
      ) : null}
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <p className="text-sm font-semibold text-zinc-700">AI Chat</p>
        <select
          value={selectedModelId ?? ""}
          onChange={(event) => setSelectedModelId(event.target.value || null)}
          disabled={isModelsLoading || modelOptions.length === 0}
          className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-300 disabled:cursor-not-allowed disabled:bg-zinc-100"
          aria-label="Select AI model"
        >
          {modelOptions.length === 0 ? (
            <option value="">
              {isModelsLoading ? "Loading models..." : "No models available"}
            </option>
          ) : (
            modelOptions.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))
          )}
        </select>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-3">
          {messages.length === 0 ? (
            <p className="text-sm text-zinc-500">
              Ask a question to get started.
            </p>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`max-w-[90%] ${message.role === "user" ? "self-end" : "self-start"
                  }`}
              >
                <div
                  className={`rounded-2xl px-4 py-2 text-sm leading-6 ${message.role === "user"
                      ? "bg-zinc-900 text-white"
                      : "bg-zinc-100 text-zinc-900"
                    }`}
                >
                  {message.role === "assistant" ? (
                    <div className="chat-markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    message.content
                  )}
                </div>
                <div className="mt-2 flex items-center gap-2 text-xs text-zinc-500">
                  <button
                    type="button"
                    onClick={() => handleCopy(message.id, message.content)}
                    className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-600 transition hover:bg-zinc-50"
                  >
                    {copiedId === message.id ? "✓" : "Copy"}
                  </button>
                </div>
              </div>
            ))
          )}
          {isSending ? (
            <div className="flex items-center gap-2 self-start rounded-2xl bg-zinc-100 px-4 py-2 text-sm text-zinc-500">
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-500" />
              Working
            </div>
          ) : null}
        </div>
      </div>

      {attachments.length > 0 ? (
        <div className="border-t border-zinc-200 px-4 py-3">
          <div className="flex flex-wrap gap-2">
            {attachments.map((attachment, index) => (
              <div
                key={`${attachment.name}-${index}`}
                className="relative flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-600"
              >
                <span className="max-w-[180px] truncate">{attachment.name}</span>
                <button
                  type="button"
                  onClick={() => removeAttachment(index)}
                  className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full border border-zinc-200 bg-white text-[10px] text-zinc-500 shadow-sm"
                  aria-label="Remove attachment"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="border-t border-zinc-200 bg-white">
        <ChatBar mode="embedded" />
      </div>
    </aside>
  );
}
