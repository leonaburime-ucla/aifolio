"use client";

import ChatSidebar from "@/features/ai-chat/react/views/components/ChatSidebar";
import { useLandingChatOrchestrator } from "@/ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator";

export default function LandingChatSidebar({
  orchestrator = useLandingChatOrchestrator,
}) {
  return <ChatSidebar chatOrchestrator={orchestrator} />;
}
