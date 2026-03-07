import { Suspense } from "react";
import CopilotSidebar from "@/features/ag-ui-chat/typescript/react/views/components/CopilotSidebar";
import CopilotChatProvider from "@/features/ag-ui-chat/typescript/react/views/providers/CopilotChatProvider";
import AgUiWorkspace from "@/features/ag-ui-chat/typescript/react/views/components/AgUiWorkspace";
import AgUiTabSwitchTool from "@/features/ag-ui-chat/typescript/react/views/components/AgUiTabSwitchTool";
import AgUiCopilotReadableContext from "@/features/ag-ui-chat/typescript/react/views/components/AgUiCopilotReadableContext";
import AgUiPageQuerySync from "@/features/ag-ui-chat/typescript/react/views/components/AgUiPageQuerySync";

export default function AgUiPage() {
  return (
    <div className="flex h-[calc(100dvh-64px)] flex-row overflow-hidden bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 overflow-y-auto py-2">
        <div className="mx-auto flex max-w-5xl flex-col gap-3 px-6">

          <details className="rounded-2xl border border-zinc-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
            <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
              What is AG-UI?
            </summary>
            <div className="mt-3 space-y-3 text-sm text-zinc-700">
              <p>
                AG-UI is a protocol for agent-to-UI actions. Instead of only returning text, an LLM can call
                structured tools that mutate the interface: switch tabs, select datasets, clear/add charts, and
                trigger page workflows.
              </p>
              <p>
                CopilotKit is the runtime bridge that registers those frontend tools and executes them safely in
                this app. It maps model tool calls to typed handlers in React so chat can control real UI state.
              </p>
              <p>
                In this workspace, chat can orchestrate multi-step flows across tabs by combining navigation and
                feature-specific tools in sequence.
              </p>
              <p>
                References:
                {" "}
                <a
                  href="https://github.com/ag-ui-protocol/ag-ui"
                  target="_blank"
                  rel="noreferrer"
                  className="underline decoration-zinc-400 underline-offset-2 hover:text-zinc-900"
                >
                  AG-UI
                </a>
                {" "}
                |
                {" "}
                <a
                  href="https://github.com/CopilotKit/CopilotKit"
                  target="_blank"
                  rel="noreferrer"
                  className="underline decoration-zinc-400 underline-offset-2 hover:text-zinc-900"
                >
                  CopilotKit
                </a>
              </p>
            </div>
          </details>
          <Suspense fallback={null}>
            <AgUiWorkspace />
          </Suspense>
        </div>
      </main>

      <div className="flex h-full w-[420px] shrink-0 flex-col overflow-y-auto">
        <CopilotChatProvider>
          <Suspense fallback={null}>
            <AgUiPageQuerySync />
          </Suspense>
          <AgUiTabSwitchTool />
          <AgUiCopilotReadableContext />
          <CopilotSidebar mode="ag-ui" />
        </CopilotChatProvider>
      </div>
    </div>
  );
}
