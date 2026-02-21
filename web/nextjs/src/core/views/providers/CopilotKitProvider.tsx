"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

type CopilotKitProviderProps = {
  children: React.ReactNode;
};

export default function CopilotKitProvider({
  children,
}: CopilotKitProviderProps) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="agentic-research">
      {children}
    </CopilotKit>
  );
}
