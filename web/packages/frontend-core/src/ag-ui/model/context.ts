import type { AgUiModelOption } from "@aifolio/contracts/entities/ag-ui";

export type { AgUiModelOption } from "@aifolio/contracts/entities/ag-ui";

export type ReadableDatasetOption = {
  id: string;
  label: string;
};

export function resolveNextAgUiSelectedModelId({
  currentSelectedModelId,
  fetchedModels,
  apiCurrentModelId,
  preferredModelId,
}: {
  currentSelectedModelId: string | null;
  fetchedModels: AgUiModelOption[];
  apiCurrentModelId: string | null;
  preferredModelId: string;
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

  if (fetchedModels.some((model) => model.id === preferredModelId)) {
    return preferredModelId;
  }

  if (apiCurrentModelId && fetchedModels.some((model) => model.id === apiCurrentModelId)) {
    return apiCurrentModelId;
  }

  return fetchedModels[0]?.id ?? null;
}

export function toReadableModelOptions(
  modelOptions: AgUiModelOption[]
): Array<{ id: string; label: string }> {
  return modelOptions.map((entry) => ({ id: entry.id, label: entry.label }));
}

export function toReadableDatasetOptions(
  datasetOptions: ReadableDatasetOption[]
): Array<{ id: string; label: string }> {
  return datasetOptions.map((entry) => ({ id: entry.id, label: entry.label }));
}
