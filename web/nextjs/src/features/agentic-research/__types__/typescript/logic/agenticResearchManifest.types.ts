import type { DatasetManifestEntry, DatasetOption } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

export type ResolveDefaultDatasetIdInput = {
  selectedDatasetId: string | null;
  datasets: DatasetManifestEntry[];
};

export type ToDatasetOptionsInput = {
  datasetManifest: DatasetManifestEntry[];
};

export type ToDatasetOptionsResult = DatasetOption[];
