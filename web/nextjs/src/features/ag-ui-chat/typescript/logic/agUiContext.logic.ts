import type { AgUiModelOption } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiModel.types";
import { AG_UI_PREFERRED_MODEL_ID } from "@/features/ag-ui-chat/typescript/config/agUiModel.config";

/**
 * AG-UI context and model-selection logic.
 *
 * Purpose:
 * - Resolve stable model selection after model option refresh.
 * - Normalize model/dataset options for Copilot-readable context.
 *
 * Layering:
 * - Pure logic module used by hooks; no React/store/runtime dependencies.
 */

export type ReadableDatasetOption = {
  id: string;
  label: string;
};

/**
 * Resolves which model id should be selected after loading model options.
 */
export function resolveNextAgUiSelectedModelId({
  currentSelectedModelId,
  fetchedModels,
  apiCurrentModelId,
}: {
  currentSelectedModelId: string | null;
  fetchedModels: AgUiModelOption[];
  apiCurrentModelId: string | null;
}): string | null {
  if (fetchedModels.length === 0) {
    return null;
  }

  const hasCurrent =
    currentSelectedModelId !== null &&
    fetchedModels.some((model) => model.id === currentSelectedModelId);
  if (hasCurrent) {
    return currentSelectedModelId;
  }

  if (fetchedModels.some((model) => model.id === AG_UI_PREFERRED_MODEL_ID)) {
    return AG_UI_PREFERRED_MODEL_ID;
  }

  if (apiCurrentModelId && fetchedModels.some((model) => model.id === apiCurrentModelId)) {
    return apiCurrentModelId;
  }

  return fetchedModels[0]?.id ?? null;
}

/**
 * Normalizes model options for Copilot-readable context.
 */
export function toReadableModelOptions(
  modelOptions: AgUiModelOption[]
): Array<{ id: string; label: string }> {
  return modelOptions.map((entry) => ({ id: entry.id, label: entry.label }));
}

/**
 * Normalizes dataset options for Copilot-readable context.
 */
export function toReadableDatasetOptions(
  datasetOptions: ReadableDatasetOption[]
): Array<{ id: string; label: string }> {
  return datasetOptions.map((entry) => ({ id: entry.id, label: entry.label }));
}
