import type {
  DatasetManifestEntry,
} from "@/features/agentic-research/__types__/typescript/agenticResearch.types";
import type {
  SetActiveDatasetErrorResponse,
  SetActiveDatasetSuccessResponse,
} from "@/features/agentic-research/__types__/typescript/ai/tools/types";

function normalizeDatasetToken(value: string): string {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function resolveDatasetId(
  datasetInput: string,
  manifest: DatasetManifestEntry[]
): string | null {
  const raw = String(datasetInput || "").trim();
  if (!raw) return null;

  const direct = manifest.find((entry) => entry.id === raw);
  if (direct) return direct.id;

  const inputNorm = normalizeDatasetToken(raw);
  if (!inputNorm) return null;

  const exact = manifest.find((entry) => {
    const idNorm = normalizeDatasetToken(entry.id);
    const labelNorm = normalizeDatasetToken(entry.label);
    return idNorm === inputNorm || labelNorm === inputNorm;
  });
  if (exact) return exact.id;

  const fuzzy = manifest.find((entry) => {
    const idNorm = normalizeDatasetToken(entry.id);
    const labelNorm = normalizeDatasetToken(entry.label);
    return (
      (idNorm.length > 0 &&
        (idNorm.includes(inputNorm) || inputNorm.includes(idNorm))) ||
      (labelNorm.length > 0 &&
        (labelNorm.includes(inputNorm) || inputNorm.includes(labelNorm)))
    );
  });
  return fuzzy?.id ?? null;
}

/**
 * AI tool helper: switch active dataset and clear stale chart state.
 */
export function handleAgenticSetActiveDataset(
  datasetInput: string,
  datasetManifest: DatasetManifestEntry[],
  clearChartsFn: () => void,
  setDatasetFn: (id: string) => void
): SetActiveDatasetSuccessResponse | SetActiveDatasetErrorResponse {
  const allowedIds = datasetManifest.map((entry) => entry.id);
  const resolvedDatasetId = resolveDatasetId(datasetInput, datasetManifest);

  if (!resolvedDatasetId || !allowedIds.includes(resolvedDatasetId)) {
    return {
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: datasetInput,
      allowed_dataset_ids: allowedIds,
    };
  }

  clearChartsFn();
  setDatasetFn(resolvedDatasetId);

  return {
    status: "ok",
    active_dataset_id: resolvedDatasetId,
  };
}
