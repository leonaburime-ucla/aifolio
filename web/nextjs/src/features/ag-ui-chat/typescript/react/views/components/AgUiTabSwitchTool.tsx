"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useRouter } from "next/navigation";
import type {
  AgUiTabSwitchArgs,
  CopilotActionParameter,
} from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotTools.types";
import { SWITCH_AG_UI_TAB_TOOL } from "@/features/ag-ui-chat/typescript/config/frontendTools.config";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import { handleSwitchAgUiTab } from "@/features/ag-ui-chat/typescript/logic/frontendTools.logic";

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
      handler: ({ tab }: AgUiTabSwitchArgs) => {
        const result = handleSwitchAgUiTab(tab);
        if (result.status === "ok") {
          setActiveTab(result.tab);
          router.push("/ag-ui");
        }
        return result;
      },
    },
    [router, setActiveTab]
  );

  return null;
}
