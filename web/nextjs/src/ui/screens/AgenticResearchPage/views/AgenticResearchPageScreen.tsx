"use client";

import dynamic from "next/dynamic";
import { useEffect } from "react";
import type { ChatOrchestrator } from "@/features/ai-chat/react/orchestrators/chatOrchestrator";
import AgenticResearchWorkspaceSurface from "@/features/agentic-research/react/views/screens/AgenticResearchWorkspaceSurface";
import type { AgenticResearchOrchestratorModel } from "@aifolio/contracts/entities/agentic-research";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/react/orchestrators/agenticResearchOrchestrator";
import { useAgenticResearchChatOrchestrator } from "@/ui/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
}

type AgenticResearchPageProps = {
  pageOrchestrator?: () => AgenticResearchOrchestratorModel;
  chatOrchestrator?: () => ChatOrchestrator;
  showChatSidebar?: boolean;
  algorithmsAccordionInitiallyOpen?: boolean;
  algorithmsAccordionTitle?: string;
  showAlgorithmsResultsCallout?: boolean;
  showAlgorithmsSamplePrompts?: boolean;
};

const ChatSidebar = dynamic(
  () => import("@/features/ai-chat/react/views/components/ChatSidebar"),
  { ssr: false }
);

export default function AgenticResearchPageScreen({
  pageOrchestrator = useAgenticResearchOrchestrator,
  chatOrchestrator = useAgenticResearchChatOrchestrator,
  showChatSidebar = true,
  algorithmsAccordionInitiallyOpen = true,
  algorithmsAccordionTitle = "ML Algorithms + Sample Prompts",
  showAlgorithmsResultsCallout = true,
  showAlgorithmsSamplePrompts = true,
}: AgenticResearchPageProps) {

  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[page-debug] agentic_research_page_mounted", {
        path: getDebugPath(),
        showChatSidebar,
        algorithmsAccordionInitiallyOpen,
      });
    }
  }, [algorithmsAccordionInitiallyOpen, showChatSidebar]);

  const content = (
    <AgenticResearchWorkspaceSurface
      pageOrchestrator={pageOrchestrator}
      algorithmsAccordionInitiallyOpen={algorithmsAccordionInitiallyOpen}
      algorithmsAccordionTitle={algorithmsAccordionTitle}
      showAlgorithmsResultsCallout={showAlgorithmsResultsCallout}
      showAlgorithmsSamplePrompts={showAlgorithmsSamplePrompts}
    />
  );

  /* When embedded (no sidebar), skip the page-level wrappers to avoid double padding */
  if (!showChatSidebar) {
    return <div className="flex flex-col gap-4">{content}</div>;
  }

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Agentic Research
          </p>
          {content}
        </div>
      </main>

      {showChatSidebar ? (
        <div className="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
          <ChatSidebar
            chatOrchestrator={chatOrchestrator}
            className="!h-full border-l-0"
          />
        </div>
      ) : null}
    </div>
  );
}
