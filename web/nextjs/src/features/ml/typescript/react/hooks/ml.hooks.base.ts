import { useState } from "react";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingProgress,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type { NumericInputSnapshot } from "@/features/ml/__types__/typescript/utils/trainingUiShared.types";
import type {
  MlTaskType,
} from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";

/**
 * Provides shared ML training UI state used by both PyTorch and TensorFlow hooks.
 * Keeps common state shape centralized so feature hooks only add model-specific fields.
 */
export function useMlTrainingUiBaseState() {
  const [targetColumn, setTargetColumn] = useState("");
  const [excludeColumnsInput, setExcludeColumnsInput] = useState<string | null>(null);
  const [dateColumnsInput, setDateColumnsInput] = useState<string | null>(null);
  const [task, setTask] = useState<MlTaskType>("auto");
  const [epochValuesInput, setEpochValuesInput] = useState("60");
  const [testSizesInput, setTestSizesInput] = useState("0.2");
  const [learningRatesInput, setLearningRatesInput] = useState("0.001");
  const [batchSizesInput, setBatchSizesInput] = useState("64");
  const [hiddenDimsInput, setHiddenDimsInput] = useState("128");
  const [numHiddenLayersInput, setNumHiddenLayersInput] = useState("2");
  const [dropoutsInput, setDropoutsInput] = useState("0.1");
  const [runSweepEnabled, setRunSweepEnabled] = useState(false);
  const [savedNumericInputs, setSavedNumericInputs] = useState<NumericInputSnapshot | null>(null);
  const [savedSweepInputs, setSavedSweepInputs] = useState<NumericInputSnapshot | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isDistilling, setIsDistilling] = useState(false);
  const [autoDistillEnabled, setAutoDistillEnabled] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({
    current: 0,
    total: 0,
  });
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [copyRunsStatus, setCopyRunsStatus] = useState<string | null>(null);
  const [optimizerStatus, setOptimizerStatus] = useState<string | null>(null);
  const [distillStatus, setDistillStatus] = useState<string | null>(null);
  const [saveDistilledModel, setSaveDistilledModel] = useState(false);
  const [isOptimalModalOpen, setIsOptimalModalOpen] = useState(false);
  const [pendingOptimalParams, setPendingOptimalParams] = useState<{
    epochs: number;
    learning_rate: number;
    test_size: number;
    batch_size: number;
    hidden_dim: number;
    num_hidden_layers: number;
    dropout: number;
  } | null>(null);
  const [pendingOptimalPrediction, setPendingOptimalPrediction] = useState<{
    metricName: string;
    metricValue: number;
  } | null>(null);
  const [isDistillMetricsModalOpen, setIsDistillMetricsModalOpen] = useState(false);
  const [distillMetrics, setDistillMetrics] = useState<TrainingMetrics | null>(null);
  const [distillModelId, setDistillModelId] = useState<string | null>(null);
  const [distillModelPath, setDistillModelPath] = useState<string | null>(null);
  const [distillComparison, setDistillComparison] = useState<DistillComparison | null>(null);

  const state: MlTrainingUiBaseState = {
    targetColumn,
    setTargetColumn,
    excludeColumnsInput,
    setExcludeColumnsInput,
    dateColumnsInput,
    setDateColumnsInput,
    task,
    setTask,
    epochValuesInput,
    setEpochValuesInput,
    testSizesInput,
    setTestSizesInput,
    learningRatesInput,
    setLearningRatesInput,
    batchSizesInput,
    setBatchSizesInput,
    hiddenDimsInput,
    setHiddenDimsInput,
    numHiddenLayersInput,
    setNumHiddenLayersInput,
    dropoutsInput,
    setDropoutsInput,
    runSweepEnabled,
    setRunSweepEnabled,
    savedNumericInputs,
    setSavedNumericInputs,
    savedSweepInputs,
    setSavedSweepInputs,
    isTraining,
    setIsTraining,
    isDistilling,
    setIsDistilling,
    autoDistillEnabled,
    setAutoDistillEnabled,
    trainingProgress,
    setTrainingProgress,
    trainingError,
    setTrainingError,
    copyRunsStatus,
    setCopyRunsStatus,
    optimizerStatus,
    setOptimizerStatus,
    distillStatus,
    setDistillStatus,
    saveDistilledModel,
    setSaveDistilledModel,
    isOptimalModalOpen,
    setIsOptimalModalOpen,
    pendingOptimalParams,
    setPendingOptimalParams,
    pendingOptimalPrediction,
    setPendingOptimalPrediction,
    isDistillMetricsModalOpen,
    setIsDistillMetricsModalOpen,
    distillMetrics,
    setDistillMetrics,
    distillModelId,
    setDistillModelId,
    distillModelPath,
    setDistillModelPath,
    distillComparison,
    setDistillComparison,
  };

  return state;
}
