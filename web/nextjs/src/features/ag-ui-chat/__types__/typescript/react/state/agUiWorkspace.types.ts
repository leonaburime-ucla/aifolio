export type AgUiWorkspaceTab = "charts" | "agentic-research" | "pytorch" | "tensorflow";

export type AgUiWorkspaceStoreState = {
  activeTab: AgUiWorkspaceTab;
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
};

export type AgUiWorkspaceStatePort = {
  activeTab: AgUiWorkspaceTab;
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
};
