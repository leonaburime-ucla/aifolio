"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useRouter } from "next/navigation";
import type {
  CopilotActionParameter,
} from "@aifolio/contracts/entities/ag-ui";
import {
  SWITCH_AG_UI_TAB_TOOL,
  handleSwitchAgUiTab,
  formatSwitchAgUiTabToolResult,
} from "@aifolio/frontend-core/ag-ui";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/react/state/adapters/agUiWorkspaceState.adapter";

const SWITCH_AG_UI_TAB_PARAMETERS: CopilotActionParameter[] = [
  {
    name: "tab",
    type: "string",
    required: true,
    description: "Tab name for /ag-ui workspace. Allowed: charts, agentic-research, pytorch, tensorflow.",
  },
];

/**
 * Purpose: AG-UI-local tool registration for switching workspace tabs.
 */
export default function AgUiTabSwitchTool() {
  const { setActiveTab } = useAgUiWorkspaceStateAdapter();
  const router = useRouter();

  useCopilotAction(
    {
      name: SWITCH_AG_UI_TAB_TOOL,
      description: "Switch the active /ag-ui workspace tab without leaving the AG-UI page.",
      parameters: SWITCH_AG_UI_TAB_PARAMETERS,
      handler: (args: Record<string, unknown>) => {
        const tab = typeof args.tab === "string" ? args.tab : "";
        const result = handleSwitchAgUiTab(tab);
        if (result.status === "ok") {
          setActiveTab(result.tab);
          router.push("/ag-ui");
        }
        return formatSwitchAgUiTabToolResult(result);
      },
    } as Parameters<typeof useCopilotAction>[0],
    [router, setActiveTab]
  );

  return null;
}
