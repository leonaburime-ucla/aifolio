/**
 * Spec: ml-training.orchestrator.spec.ts
 * Version: 1.1.0
 */
export const ML_TRAINING_ORCH_SPEC_VERSION = "1.1.0";

export const mlTrainingOrchestratorSpec = {
  id: "ml-training.orchestrator",
  version: ML_TRAINING_ORCH_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  units: [
    "useMlDatasetOrchestrator",
    "runPytorchTraining",
    "runPytorchDistillation",
    "runTensorflowTraining",
    "runTensorflowDistillation",
  ],
  inputContract: {
    useMlDatasetOrchestrator: "{useDatasetState?,loadDatasetOptions?,loadDatasetRows?}",
    runPytorchTraining: "{datasetId,targetColumn,task,excludeColumns,dateColumns,combinations} + deps",
    runTensorflowTraining: "{datasetId,targetColumn,task,trainingMode,isLinearBaselineMode,excludeColumns,dateColumns,combinations} + deps",
  },
  outputContract: {
    useMlDatasetOrchestrator: "{datasetOptions,selectedDatasetId,setSelectedDatasetId,isLoading,error,tableRows,tableColumns,rowCount,totalRowCount}",
    runPytorchDistillation: "{status:'ok',metrics,modelId,modelPath,distilledRun} | {status:'error',error}",
    runTensorflowDistillation: "{status:'ok',metrics,modelId,modelPath,distilledRun} | {status:'error',error}",
  },
  behaviorRules: [
    "Training loops emit one prependTrainingRun call per combination iteration.",
    "Training loops call onProgress after each trainModel response.",
    "No orchestrator imports app/core consumer screens.",
  ],
} as const;
