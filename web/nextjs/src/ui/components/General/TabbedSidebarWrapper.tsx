"use client";

import { useMemo, useState } from "react";

type SidebarTab = {
  id: string;
  label: string;
  content: React.ReactNode;
};

type TabbedSidebarWrapperProps = {
  tabs: SidebarTab[];
  defaultTabId?: string;
  className?: string;
};

export default function TabbedSidebarWrapper({
  tabs,
  defaultTabId,
  className = "",
}: TabbedSidebarWrapperProps) {
  const firstTabId = tabs[0]?.id;
  const initialTabId = defaultTabId && tabs.some((tab) => tab.id === defaultTabId)
    ? defaultTabId
    : firstTabId;
  const [activeTabId, setActiveTabId] = useState<string | undefined>(initialTabId);

  const activeTab = useMemo(
    () => tabs.find((tab) => tab.id === activeTabId) ?? tabs[0],
    [tabs, activeTabId]
  );

  return (
    <aside className={`flex h-full min-h-0 w-full flex-col border-l border-zinc-200 bg-white ${className}`}>
      <div className="border-b border-zinc-200 bg-zinc-50 p-1">
        <div className="grid grid-cols-2 gap-1">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTab?.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTabId(tab.id)}
                className={`rounded-md px-3 py-2 text-xs font-semibold transition ${
                  isActive
                    ? "bg-white text-zinc-900 shadow-sm"
                    : "text-zinc-600 hover:bg-zinc-100"
                }`}
                aria-pressed={isActive}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        {activeTab?.content ?? null}
      </div>
    </aside>
  );
}
