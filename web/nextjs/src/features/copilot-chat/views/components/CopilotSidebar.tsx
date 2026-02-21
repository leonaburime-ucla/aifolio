"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import CopilotAssistantMessage from "@/features/copilot-chat/views/components/CopilotAssistantMessage";
import CopilotAssistantMessageLegacy from "@/features/copilot-chat/views/components/CopilotAssistantMessageLegacy";
import CopilotFrontendTools from "@/features/copilot-chat/views/components/CopilotFrontendTools";

/**
 * Sidebar chat component for the Copilot/AG-UI feature slice.
 *
 * It is intentionally UI-only; networking/runtime setup lives in:
 * - `views/providers/CopilotChatProvider.tsx`
 * - `adapters/copilotRuntime.adapter.ts`
 */
type CopilotSidebarProps = {
  mode?: "legacy" | "ag-ui";
  className?: string;
  frontendTools?: React.ReactNode;
};

export default function CopilotSidebar({
  mode = "legacy",
  className = "",
  frontendTools,
}: CopilotSidebarProps) {
  const isAgUiMode = mode === "ag-ui";
  return (
    <aside className={`flex h-full w-full flex-col border-l border-zinc-200 bg-white ${className}`}>
      {isAgUiMode ? (frontendTools ?? <CopilotFrontendTools />) : null}
      <CopilotChat
        className="h-full"
        labels={{ title: "AI Chat", initial: "Ask a question to get started." }}
        AssistantMessage={isAgUiMode ? CopilotAssistantMessage : CopilotAssistantMessageLegacy}
      />
    </aside>
  );
}
