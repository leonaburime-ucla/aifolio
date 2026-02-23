/**
 * Spec: ml-training.state.spec.ts
 * Version: 1.1.0
 */
export const ML_TRAINING_STATE_SPEC_VERSION = "1.1.0";

export const mlTrainingStateSpec = {
  id: "ml-training.state",
  version: ML_TRAINING_STATE_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  stores: ["mlDatasetStore", "useMlTrainingRunsStore"],
  ports: ["useMlDatasetStateAdapter", "useMlTrainingRunsAdapter"],
  invariants: [
    "prependTrainingRun prepends to the beginning of trainingRuns array.",
    "clearTrainingRuns resets trainingRuns to an empty array.",
    "mlDatasetStore keeps cache entries keyed by dataset id.",
  ],
  transitions: [
    {
      action: "loadManifest:success",
      before: "manifestLoaded=false",
      after: "manifestLoaded=true and selectedDatasetId fallback applied",
    },
    {
      action: "setDatasetCacheEntry",
      before: "datasetCache may have existing keys",
      after: "datasetCache[datasetId] replaced with provided entry",
    },
  ],
} as const;
