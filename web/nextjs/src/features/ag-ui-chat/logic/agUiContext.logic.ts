import { AG_UI_PREFERRED_MODEL_ID } from "@aifolio/frontend-core/ag-ui";
import {
  type AgUiModelOption,
  resolveNextAgUiSelectedModelId as resolveNextAgUiSelectedModelIdCore,
  toReadableModelOptions,
  toReadableDatasetOptions,
} from "@aifolio/frontend-core/ag-ui";

export type { ReadableDatasetOption } from "@aifolio/frontend-core/ag-ui";
export { toReadableModelOptions, toReadableDatasetOptions };

/**
 * Resolves the selected AG-UI model using the app's preferred default.
 *
 * @param currentSelectedModelId Current model id in local state.
 * @param fetchedModels Models returned by the API.
 * @param apiCurrentModelId Current model id reported by the API.
 * @returns Next selected model id, or null when no model is available.
 * @complexity O(n) time over fetched models and O(1) space.
 * @overallScore 100
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
  return resolveNextAgUiSelectedModelIdCore({
    currentSelectedModelId,
    fetchedModels,
    apiCurrentModelId,
    preferredModelId: AG_UI_PREFERRED_MODEL_ID,
  });
}
