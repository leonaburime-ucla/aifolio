"use client";

import ChatSidebar from "@/features/ai/views/components/ChatSidebar";
import { useLandingChatOrchestrator } from "@/features/ai/orchestrators/landingChatOrchestrator";

export default function LandingChatSidebar() {
  return <ChatSidebar chatOrchestrator={useLandingChatOrchestrator} />;
}

