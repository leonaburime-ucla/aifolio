import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

let activeTab: "pytorch" | "tensorflow" | "charts" | "agentic-research" = "pytorch";
const registeredActionNames: string[] = [];

vi.mock("@copilotkit/react-core", () => ({
  useCopilotAction: (config: { name?: string; disabled?: boolean }) => {
    if (config?.disabled) {
      return;
    }
    if (typeof config?.name === "string") {
      registeredActionNames.push(config.name);
    }
  },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

vi.mock("@/features/recharts/typescript/react/ai/state/adapters/chartActions.adapter", () => ({
  useCopilotChartActionsAdapter: () => ({
    addChartSpec: vi.fn(),
    clearChartSpecs: vi.fn(),
  }),
}));

vi.mock("@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter", () => ({
  useAgenticResearchChartActionsAdapter: () => ({
    addChartSpec: vi.fn(),
    clearChartSpecs: vi.fn(),
  }),
}));

vi.mock("@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter", () => ({
  useAgUiWorkspaceStateAdapter: () => ({
    activeTab,
    setActiveTab: vi.fn(),
  }),
}));

vi.mock("@/features/ag-ui-chat/typescript/react/state/zustand/agUiWorkspaceStore", () => ({
  useAgUiWorkspaceStore: {
    getState: () => ({
      activeTab,
    }),
  },
}));

import { useCopilotFrontendTools } from "@/features/ag-ui-chat/typescript/api/hooks/useCopilotFrontendTools.hooks";

describe("useCopilotFrontendTools registration", () => {
  beforeEach(() => {
    registeredActionNames.length = 0;
    activeTab = "pytorch";
  });

  it("does not register TensorFlow-specific ML tools while the PyTorch tab is active", () => {
    activeTab = "pytorch";

    renderHook(() => useCopilotFrontendTools());

    expect(registeredActionNames).toContain("set_active_ml_form_fields");
    expect(registeredActionNames).toContain("set_pytorch_form_fields");
    expect(registeredActionNames).not.toContain("set_tensorflow_form_fields");
    expect(registeredActionNames).not.toContain("start_tensorflow_training_runs");
  });

  it("does not register PyTorch-specific ML tools while the TensorFlow tab is active", () => {
    activeTab = "tensorflow";

    renderHook(() => useCopilotFrontendTools());

    expect(registeredActionNames).toContain("set_active_ml_form_fields");
    expect(registeredActionNames).toContain("set_tensorflow_form_fields");
    expect(registeredActionNames).not.toContain("set_pytorch_form_fields");
    expect(registeredActionNames).not.toContain("start_pytorch_training_runs");
  });
});
