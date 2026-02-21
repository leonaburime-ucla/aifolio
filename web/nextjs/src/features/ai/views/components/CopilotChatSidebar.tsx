"use client";

import { CopilotChat } from "@copilotkit/react-ui";

export default function CopilotChatSidebar() {
  return (
    <aside className="flex h-[calc(100vh-64px)] w-full flex-col border-l border-zinc-200 bg-white">
      <CopilotChat
        className="h-full"
        labels={{ title: "AI Chat", initial: "Ask a question to get started." }}
      />
    </aside>
  );
}
