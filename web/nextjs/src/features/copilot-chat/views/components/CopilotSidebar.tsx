"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import CopilotAssistantMessage from "@/features/copilot-chat/views/components/CopilotAssistantMessage";
import CopilotAssistantMessageLegacy from "@/features/copilot-chat/views/components/CopilotAssistantMessageLegacy";

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
type CopilotSidebarProps = {
  mode?: "legacy" | "ag-ui";
  className?: string;
};

export default function CopilotSidebar({
  mode = "legacy",
  className = "",
}: CopilotSidebarProps) {
  const isAgUiMode = mode === "ag-ui";
  return (
    <aside className={`flex h-full w-full flex-col border-l border-zinc-200 bg-white ${className}`}>
      <CopilotChat
        className="h-full"
        labels={{ title: "AI Chat", initial: "Ask a question to get started." }}
        AssistantMessage={isAgUiMode ? CopilotAssistantMessage : CopilotAssistantMessageLegacy}
      />
    </aside>
  );
}
