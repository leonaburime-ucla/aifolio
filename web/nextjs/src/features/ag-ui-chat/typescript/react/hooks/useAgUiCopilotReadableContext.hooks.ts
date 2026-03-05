"use client";

import { useCopilotReadable } from "@copilotkit/react-core";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import { useAgUiModelStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiModelState.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter";
import { useMlDatasetStateAdapter } from "@/features/ml/typescript/react/state/adapters/mlDataset.adapter";
import {
  toReadableDatasetOptions,
  toReadableModelOptions,
} from "@/features/ag-ui-chat/typescript/logic/agUiContext.logic";

/**
 * Publishes AG-UI and related dataset context values to Copilot-readable state.
 */
export function useAgUiCopilotReadableContext(): void {
  const { activeTab } = useAgUiWorkspaceStateAdapter();
  const { selectedModelId, modelOptions } = useAgUiModelStateAdapter();
  const { state } = useAgenticResearchStateAdapter();
  const { state: mlState } = useMlDatasetStateAdapter();

  useCopilotReadable({
    description: "ag_ui_active_tab",
    value: activeTab,
  });

  useCopilotReadable({
    description: "ag_ui_selected_model_id",
    value: selectedModelId ?? "",
  });

  useCopilotReadable({
    description: "ag_ui_model_options",
    value: toReadableModelOptions(modelOptions),
  });

  useCopilotReadable({
    description: "agentic_research_selected_dataset_id",
    value: state.selectedDatasetId ?? "",
  });

  useCopilotReadable({
    description: "agentic_research_dataset_options",
    value: toReadableDatasetOptions(state.datasetManifest),
  });

  useCopilotReadable({
    description: "ml_selected_dataset_id",
    value: mlState.selectedDatasetId ?? "",
  });

  useCopilotReadable({
    description: "ml_dataset_options",
    value: toReadableDatasetOptions(mlState.datasetOptions),
  });
}
