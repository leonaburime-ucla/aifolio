"use client";

import { useAgUiWorkspaceOrchestrator } from "@/features/ag-ui-chat/react/orchestrators/agUiWorkspace.orchestrator";
import { AG_UI_WORKSPACE_SURFACES } from "@/features/ag-ui-chat/react/registry/agUiWorkspace.registry";
import AgUiToolsModal from "@/features/ag-ui-chat/react/views/components/AgUiToolsModal";

/**
 * Unified AG-UI workspace that hosts all major analysis and ML screens.
 */
export default function AgUiWorkspace() {
  const { activeTab, tabs, handleTabClick } = useAgUiWorkspaceOrchestrator();
  const ActiveSurface = AG_UI_WORKSPACE_SURFACES[activeTab];

  return (
    <>
      <div className="sticky top-0 z-20 rounded-2xl border border-zinc-200 bg-white/90 p-2 shadow-sm backdrop-blur-sm">
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => handleTabClick(tab.id)}
              className={`rounded-xl px-3 py-2 text-sm font-medium transition ${activeTab === tab.id
                ? "bg-zinc-900 text-white"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200"
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="my-0.5">
        <AgUiToolsModal activeTab={activeTab} />
      </div>
      <ActiveSurface />
    </>
  );
}
