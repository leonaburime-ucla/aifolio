import CopilotChatProvider from "@/features/copilot-chat/views/providers/CopilotChatProvider";

export default function AgUiLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <CopilotChatProvider>{children}</CopilotChatProvider>;
}
