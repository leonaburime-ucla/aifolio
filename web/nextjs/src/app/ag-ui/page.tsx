import CopilotSidebar from "@/features/copilot-chat/views/components/CopilotSidebar";
import CopilotChatProvider from "@/features/copilot-chat/views/providers/CopilotChatProvider";
import LandingCharts from "@/core/views/screens/LandingCharts";

export default function AgUiPage() {
  return (
    <div className="flex h-[calc(100dvh-64px)] flex-row overflow-hidden bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 overflow-y-auto py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Agentic UI (AG-UI Tool Call Mode)
          </p>
          <details className="rounded-2xl border border-zinc-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm" open>
            <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
              What This AG-UI Page Is
            </summary>
            <div className="mt-3 space-y-3 text-sm text-zinc-700">
              <p>
                This page uses AG-UI + CopilotKit for frontend tool calls, so the agent can trigger UI actions
                directly from chat workflows.
              </p>
              <p>
                GitHub:
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
              <p>
                You can create charts here similar to the analysis-style charts in
                {" "}
                <code>/agentic-research</code>.
              </p>
            </div>
          </details>
          <LandingCharts />
        </div>
      </main>

      <div className="h-full w-[420px] shrink-0 overflow-hidden">
        <CopilotChatProvider>
          <CopilotSidebar mode="ag-ui" />
        </CopilotChatProvider>
      </div>
    </div>
  );
}
