"use client";

import { useCopilotChatInternal } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import type { CopilotSidebarProps } from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotSidebar.types";
import AgUiModelSelector from "@/features/ag-ui-chat/typescript/react/views/components/AgUiModelSelector";
import CopilotAssistantMessage from "@/features/ag-ui-chat/typescript/react/views/components/CopilotAssistantMessage";
import CopilotAssistantMessageLegacy from "@/features/ag-ui-chat/typescript/react/views/components/CopilotAssistantMessageLegacy";
import { useCopilotMessageStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/copilotMessageStore";

/**
 * Sidebar chat component for the Copilot/AG-UI feature slice.
 *
 * It is intentionally UI-only; networking/runtime setup lives in:
 * - `views/providers/CopilotChatProvider.tsx`
 * - `views/providers/CopilotEffectsProvider.tsx` (for invisible side-effects)
 * - `adapters/copilotRuntime.adapter.ts`
 *
 * Note: The CopilotEffectsProvider should wrap this component (or its parent)
 * to enable frontend tools, chart bridge, and message persistence.
 * Previously these were inline "invisible components" (CopilotFrontendTools, etc.)
 * that returned null.
 */
export default function CopilotSidebar({
  mode = "legacy",
  className = "",
}: CopilotSidebarProps) {
  const isAgUiMode = mode === "ag-ui";
  const { setMessages } = useCopilotChatInternal();
  const clearPersistedMessages = useCopilotMessageStore((state) => state.clearMessages);

  function handleNewChat() {
    setMessages([] as never[]);
    clearPersistedMessages();
  }

  return (
    <aside className={`flex h-full w-full flex-col border-l border-zinc-200 bg-white ${className}`}>
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <p className="text-sm font-semibold text-zinc-700">AI Chat</p>
        <div className="flex items-center gap-2">
          {isAgUiMode ? (
            <button
              type="button"
              onClick={handleNewChat}
              className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-50"
            >
              New Chat
            </button>
          ) : null}
          <AgUiModelSelector />
        </div>
      </div>
      <CopilotChat
        className="min-h-0 flex-1 overflow-y-auto"
        labels={{ title: "AI Chat", initial: "Ask a question to get started." }}
        AssistantMessage={isAgUiMode ? CopilotAssistantMessage : CopilotAssistantMessageLegacy}
      />
    </aside>
  );
}
