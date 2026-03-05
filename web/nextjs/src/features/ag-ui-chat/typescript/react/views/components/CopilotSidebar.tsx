"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import type { CopilotSidebarProps } from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotSidebar.types";
import AgUiModelSelector from "@/features/ag-ui-chat/typescript/react/views/components/AgUiModelSelector";
import CopilotAssistantMessage from "@/features/ag-ui-chat/typescript/react/views/components/CopilotAssistantMessage";
import CopilotAssistantMessageLegacy from "@/features/ag-ui-chat/typescript/react/views/components/CopilotAssistantMessageLegacy";

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
  return (
    <aside className={`flex h-full w-full flex-col border-l border-zinc-200 bg-white ${className}`}>
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <p className="text-sm font-semibold text-zinc-700">AI Chat</p>
        <AgUiModelSelector />
      </div>
      <CopilotChat
        className="h-full"
        labels={{ title: "AI Chat", initial: "Ask a question to get started." }}
        AssistantMessage={isAgUiMode ? CopilotAssistantMessage : CopilotAssistantMessageLegacy}
      />
    </aside>
  );
}
