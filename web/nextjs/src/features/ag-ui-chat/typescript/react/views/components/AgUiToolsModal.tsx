"use client";

import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";
import { getAgUiToolsForTab } from "@/features/ag-ui-chat/typescript/logic/agUiToolsCatalog.logic";

/**
 * AG-UI tools disclosure control.
 *
 * Renders one button and one modal that list currently available tools
 * for the active AG-UI workspace tab.
 *
 * The modal is rendered via a React Portal (`document.body`) so it is
 * never clipped by ancestor `overflow` or z-index stacking contexts.
 */
export default function AgUiToolsModal({ activeTab }: { activeTab: AgUiWorkspaceTab }) {
  const [isOpen, setIsOpen] = useState(false);
  const tools = useMemo(() => getAgUiToolsForTab(activeTab), [activeTab]);
  const tabLabel = activeTab === "agentic-research" ? "Agentic Research" : activeTab;

  return (
    <>
      <div className="flex items-center justify-between">

        <button
          type="button"
          onClick={() => setIsOpen(true)}
          className="rounded-md border border-emerald-600 bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-emerald-700"
        >
          Show Tools
        </button>
      </div>

      {isOpen
        ? createPortal(
          <div
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40 p-4"
            onClick={(e) => {
              if (e.target === e.currentTarget) setIsOpen(false);
            }}
            role="dialog"
            aria-modal="true"
          >
            <div className="w-full max-w-2xl rounded-xl border border-zinc-200 bg-white shadow-2xl">
              <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
                <h3 className="text-sm font-semibold text-zinc-900">
                  Available Tools — {tabLabel}
                </h3>
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="rounded-md border border-zinc-300 px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-50"
                  aria-label="Close tools modal"
                >
                  ✕
                </button>
              </div>
              <div className="max-h-[60vh] overflow-y-auto px-4 py-3">
                <p className="text-xs text-zinc-600">
                  Tools are callable actions the model can invoke for this page to perform
                  structured UI operations.
                </p>
                <ul className="mt-3 space-y-2 text-sm">
                  {tools.map((tool) => (
                    <li key={tool.name} className="rounded-md border border-zinc-200 px-3 py-2">
                      <p className="font-mono text-xs text-zinc-800">{tool.name}</p>
                      <p className="text-xs text-zinc-600">{tool.summary}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>,
          document.body,
        )
        : null}
    </>
  );
}
