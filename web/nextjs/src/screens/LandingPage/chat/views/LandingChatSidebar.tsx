"use client";

import ChatSidebar from "@/features/ai-chat/typescript/react/views/components/ChatSidebar";
import { useLandingChatOrchestrator } from "@/screens/LandingPage/chat/orchestrators/landingChatOrchestrator";

export default function LandingChatSidebar({
  orchestrator = useLandingChatOrchestrator
}) {
  return <ChatSidebar chatOrchestrator={orchestrator} />;
}
