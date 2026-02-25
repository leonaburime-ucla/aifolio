"use client";

import { useState } from "react";
import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import DataTable from "@/core/views/components/Datatable/DataTable";
import {
  usePytorchTrainingIntegration,
  type PytorchIntegrationArgs,
  type PytorchTrainingMode,
} from "@/features/ml/hooks/usePytorchTraining.hooks";
import { FieldHelp } from "@/features/ml/views/components/FieldHelp";
import { TrainingRunsSection } from "@/features/ml/views/components/TrainingRunsSection";
import {
  DistillMetricsModal,
  OptimalParamsModal,
} from "@/features/ml/views/components/MlTrainingModals";
import {
  ModelPreviewModal,
} from "@/features/ml/views/components/ModelPreviewModal";
import {
  runPytorchDistillation,
  runPytorchTraining,
} from "@/features/ml/orchestrators/pytorchTraining.orchestrator";

type ModeExplainer = {
  what: string;
  why: string;
  distillationNote: string;
};

const PYTORCH_MODE_EXPLAINERS: Record<PytorchTrainingMode, ModeExplainer> = {
  mlp_dense: {
    what:
      "Neural Net (dense network): learns nonlinear feature interactions using hidden layers with backpropagation.",
    why:
      "Flexible general-purpose model that captures complex interactions a linear baseline can miss, and works well as your default deep tabular learner.",
    distillationNote: "Supported.",
  },
  linear_glm_baseline: {
    what:
      "Linear/GLM baseline: a single linear head (logistic or linear regression) with no hidden stack.",
    why:
      "Fastest and easiest to interpret, making it ideal for benchmarking and sanity checks before moving to more complex models.",
    distillationNote: "Supported.",
  },
  tabresnet: {
    what:
      "TabResNet (Residual MLP): a dense network with residual skip connections between hidden blocks.",
    why:
      "Supports deeper tabular networks with better gradient flow and training stability than a plain dense stack.",
    distillationNote: "Supported.",
  },
  imbalance_aware: {
    what:
      "Imbalance-aware classifier: neural net training with class-weighted loss so minority classes matter more.",
    why:
      "Useful when target classes are skewed (for example 95/5 splits).",
    distillationNote:
      "Not supported yet because current distillation flow assumes standard unweighted teacher objectives.",
  },
  calibrated_classifier: {
    what:
      "Calibrated classifier (label-smoothed): classification training with label smoothing to reduce overconfident probabilities.",
    why:
      "Produces more conservative probability outputs that can improve confidence calibration in production decisions.",
    distillationNote:
      "Not supported yet because the current distillation objective does not preserve calibration-specific behavior.",
  },
  tree_teacher_distillation: {
    what:
      "Tree-teacher distillation: first trains a tree ensemble teacher, then trains a compact neural student to mimic the teacher and the target labels.",
    why:
      "Combines strong tabular tree patterns with a deployable neural student.",
    distillationNote:
      "This mode already includes teacher-student training, so separate post-run distillation is not supported.",
  },
};

type PyTorchPageProps = {
  orchestrator?: (args?: PytorchIntegrationArgs) => ReturnType<typeof usePytorchTrainingIntegration>;
};

/** PyTorch training page wired through the Orc-BASH integration hook. */
export default function PyTorchPage({
  orchestrator = usePytorchTrainingIntegration,
}: PyTorchPageProps) {
  const [isModelPreviewOpen, setIsModelPreviewOpen] = useState(false);
  const {
    datasetOptions,
    selectedDatasetId,
    isLoading,
    error,
    tableRows,
    tableColumns,
    rowCount,
    totalRowCount,
    trainingMode,
    setTrainingMode,
    isLinearBaselineMode,
    isStopRequested,
    targetColumn,
    setTargetColumn,
    resolvedExcludeColumnsInput,
    setExcludeColumnsInput,
    resolvedDateColumnsInput,
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
    toggleRunSweep,
    reloadSweepValues,
    isTraining,
    isDistilling,
    autoDistillEnabled,
    setAutoDistillEnabled,
    distillingTeacherKey,
    distilledByTeacher,
    trainingProgress,
    trainingError,
    plannedRunCount,
    epochsValidation,
    testSizesValidation,
    learningRatesValidation,
    batchSizesValidation,
    hiddenDimsValidation,
    numHiddenLayersValidation,
    dropoutsValidation,
    defaults,
    onDatasetChange,
    onTrainClick,
    onFindOptimalParamsClick,
    onApplyOptimalParams,
    onStopTrainingRuns,
    onDistillFromRun,
    onSeeDistilledFromRun,
    isDistillationSupportedForRun,
    trainingRuns,
    copyRunsStatus,
    onCopyTrainingRuns,
    clearTrainingRuns,
    completedRuns,
    optimizerStatus,
    distillStatus,
    isOptimalModalOpen,
    setIsOptimalModalOpen,
    pendingOptimalParams,
    pendingOptimalPrediction,
    isDistillMetricsModalOpen,
    setIsDistillMetricsModalOpen,
    distillMetrics,
    distillModelId,
    distillModelPath,
    distillComparison,
  } = orchestrator({
    runTraining: runPytorchTraining,
    runDistillation: runPytorchDistillation,
  });

  return (
    <div className="flex min-h-screen flex-row bg-white text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Machine Learning with PyTorch
          </p>
          <div className="mt-2 flex max-w-xl flex-col gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Dataset (CSV/XLS/XLSX)
            </p>
            <CsvDatasetCombobox
              options={datasetOptions}
              selectedId={selectedDatasetId}
              onChange={onDatasetChange}
              emptyMessage={
                error ?? (isLoading ? "Loading datasets..." : "No dataset found.")
              }
            />
            {error ? (
              <p className="text-xs text-red-600">{error}</p>
            ) : null}
          </div>

          <section className="rounded-xl border border-zinc-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Training Algorithm
            </p>
            <div className="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span>Select the machine learning architecture to run for this dataset.</span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={trainingMode}
                  onChange={(event) => setTrainingMode(event.target.value as PytorchTrainingMode)}
                >
                  <option value="linear_glm_baseline">linear/glm baseline</option>
                  <option value="mlp_dense">neural net (dense)</option>
                  <option value="tabresnet">tabresnet (residual mlp)</option>
                  <option value="imbalance_aware">imbalance-aware classifier</option>
                  <option value="calibrated_classifier">calibrated classifier</option>
                  <option value="tree_teacher_distillation">tree-teacher distillation</option>
                </select>
                <div className="mt-1 flex items-center gap-2">
                  <button
                    type="button"
                    className="w-fit rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white"
                    onClick={() => setIsModelPreviewOpen(true)}
                  >
                    Show Model
                  </button>
                </div>
              </label>
              <div className="md:col-span-2 rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-900">
                <p>
                  <span className="font-semibold">What it is:</span>{" "}
                  {PYTORCH_MODE_EXPLAINERS[trainingMode].what}
                </p>
                <p className="mt-1">
                  <span className="font-semibold">Why it&apos;s unique:</span>{" "}
                  {PYTORCH_MODE_EXPLAINERS[trainingMode].why}
                </p>
                <p className="mt-1">
                  <span className="font-semibold">Distillation Note:</span>{" "}
                  {PYTORCH_MODE_EXPLAINERS[trainingMode].distillationNote}
                </p>
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-zinc-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Train PyTorch Model
            </p>
            <div className="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Target Column
                  <FieldHelp text="Prediction target (label). This column is removed from model inputs and is what the model learns to predict." />
                </span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={targetColumn}
                  onChange={(event) => setTargetColumn(event.target.value)}
                >
                  <option value="">
                    {defaults.targetColumn || "Select target column"}
                  </option>
                  {tableColumns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Task
                  <FieldHelp text="auto infers classification vs regression from target values. Set explicitly when auto inference might be ambiguous." />
                </span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={task}
                  onChange={(event) =>
                    setTask(event.target.value as "classification" | "regression" | "auto")
                  }
                >
                  <option value="auto">auto</option>
                  <option value="classification">classification</option>
                  <option value="regression">regression</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Epoch Values
                  <FieldHelp text="Number of full passes over training data. Higher can improve fit but may overfit. Range: 1-500." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={epochValuesInput}
                  onChange={(event) => setEpochValuesInput(event.target.value)}
                  placeholder="e.g. 10,20,50,100,200"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Batch Sizes
                  <FieldHelp text="Rows processed per optimizer step. Larger batches are faster but can generalize differently. Range: 1-200." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={batchSizesInput}
                  onChange={(event) => setBatchSizesInput(event.target.value)}
                  placeholder="e.g. 32,64,128"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Learning Rates
                  <FieldHelp text="Optimizer step size. Too high can diverge, too low can train slowly. Valid range: >0 and <=1." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={learningRatesInput}
                  onChange={(event) => setLearningRatesInput(event.target.value)}
                  placeholder="e.g. 0.001,0.0005"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Test Sizes
                  <FieldHelp text="Fraction held out for evaluation. Example 0.2 means 20% test split. Valid range: >0 and <1." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={testSizesInput}
                  onChange={(event) => setTestSizesInput(event.target.value)}
                  placeholder="e.g. 0.2,0.3"
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Hidden Dims
                  <FieldHelp text="Width of each hidden layer in the deep branch. Larger values increase model capacity and cost. Range: 8-500." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={hiddenDimsInput}
                  onChange={(event) => setHiddenDimsInput(event.target.value)}
                  placeholder="e.g. 128,256"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Hidden Layers
                  <FieldHelp text="Number of hidden layers in the deep branch. More layers can model complex patterns but may overfit. Range: 1-15." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={numHiddenLayersInput}
                  onChange={(event) => setNumHiddenLayersInput(event.target.value)}
                  placeholder="e.g. 2,3,4"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Dropouts
                  <FieldHelp text="Dropout probability per hidden layer (0 to 0.9). Helps regularization; too high can underfit." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={dropoutsInput}
                  onChange={(event) => setDropoutsInput(event.target.value)}
                  placeholder="e.g. 0.1,0.2"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Exclude Columns
                  <FieldHelp text="Columns to drop from training features (for example IDs) as they are simply noise. Comma-separated list." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={resolvedExcludeColumnsInput}
                  onChange={(event) => setExcludeColumnsInput(event.target.value)}
                  placeholder="e.g. customerID,Order,PID"
                />
                <span className="text-[11px] text-zinc-500">
                  Preloaded: {defaults.excludeColumns.length > 0 ? defaults.excludeColumns.join(", ") : "(none)"}
                </span>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Date Columns
                  <FieldHelp text="Columns parsed as dates and expanded into engineered numeric features (month/day/week/cyclical terms)." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={resolvedDateColumnsInput}
                  onChange={(event) => setDateColumnsInput(event.target.value)}
                  placeholder="e.g. Date"
                />
                <span className="text-[11px] text-zinc-500">
                  Preloaded: {defaults.dateColumns.length > 0 ? defaults.dateColumns.join(", ") : "(none)"}
                </span>
              </label>
            </div>
            <div className="mt-2 grid max-w-3xl grid-cols-1 gap-1 text-xs text-zinc-500 md:grid-cols-2">
              <p className={epochsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Epochs: {epochsValidation.ok ? `${epochsValidation.values.join(", ")}` : epochsValidation.error}
              </p>
              <p className={batchSizesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Batch sizes: {batchSizesValidation.ok ? `${batchSizesValidation.values.join(", ")}` : batchSizesValidation.error}
              </p>
              <p className={learningRatesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Learning rates: {learningRatesValidation.ok ? `${learningRatesValidation.values.join(", ")}` : learningRatesValidation.error}
              </p>
              <p className={testSizesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Test sizes: {testSizesValidation.ok ? `${testSizesValidation.values.join(", ")}` : testSizesValidation.error}
              </p>
              <p className={isLinearBaselineMode || hiddenDimsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden dims: {isLinearBaselineMode ? "n/a (linear baseline)" : hiddenDimsValidation.ok ? `${hiddenDimsValidation.values.join(", ")}` : hiddenDimsValidation.error}
              </p>
              <p className={isLinearBaselineMode || numHiddenLayersValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden layers: {isLinearBaselineMode ? "n/a (linear baseline)" : numHiddenLayersValidation.ok ? `${numHiddenLayersValidation.values.join(", ")}` : numHiddenLayersValidation.error}
              </p>
              <p className={isLinearBaselineMode || dropoutsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Dropouts: {isLinearBaselineMode ? "n/a (linear baseline)" : dropoutsValidation.ok ? `${dropoutsValidation.values.join(", ")}` : dropoutsValidation.error}
              </p>
            </div>
            {/* Primary Action */}
            <div className="mt-5 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  className="rounded-md bg-zinc-900 px-6 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-400 disabled:shadow-none"
                  onClick={onTrainClick}
                  disabled={isTraining || isDistilling || !selectedDatasetId || plannedRunCount === 0}
                >
                  {isTraining
                    ? `Training ${trainingProgress.current}/${trainingProgress.total}...`
                    : "Train Model"}
                </button>
                <div className="flex flex-col gap-0.5">
                  <p className="text-xs text-zinc-500">
                    Dataset: <code>{selectedDatasetId ?? "none"}</code>
                  </p>
                  <p className="text-xs font-semibold text-red-600">
                    Planned runs: {plannedRunCount}
                  </p>
                </div>
              </div>
            </div>

            {/* Optional Settings Divider */}
            <div className="relative mb-6 mt-8">
              <div className="absolute inset-0 flex items-center" aria-hidden="true">
                <div className="w-full border-t border-zinc-200" />
              </div>
              <div className="relative flex justify-center">
                <span className="bg-white px-3 text-[11px] font-semibold uppercase tracking-widest text-zinc-400">
                  Optional Settings
                </span>
              </div>
            </div>

            {/* Optional Settings Grid */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* Bayesian Optimization */}
              <div className="flex flex-col">
                <p className="mb-1 text-sm font-semibold text-zinc-700">Bayesian Optimization</p>
                <p className="text-xs text-zinc-600">
                  <span className="font-semibold text-zinc-700">What is it:</span> A method for optimizing expensive black-box functions by using a probabilistic model to choose promising parameter settings.
                </p>
                <p className="mb-3 mt-1 text-xs text-zinc-600">
                  <span className="font-semibold text-zinc-700">How it works:</span> Uses completed runs to suggest the next promising hyperparameter combination. Requires at least 5 completed runs for the specific algorithm.
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 disabled:cursor-not-allowed disabled:text-zinc-400 disabled:shadow-none"
                    onClick={onFindOptimalParamsClick}
                    disabled={isTraining || isDistilling || completedRuns.length < 5}
                  >
                    Find Optimal Params
                  </button>
                  {optimizerStatus ? (
                    <span className="text-xs text-zinc-500">{optimizerStatus}</span>
                  ) : null}
                </div>
              </div>

              {/* Sweep & Distillation */}
              <div className="flex flex-col gap-5 border-zinc-200 md:border-l md:pl-6">
                <div>
                  <label className="inline-flex items-center gap-2 text-sm font-medium text-zinc-700">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-zinc-900"
                      checked={runSweepEnabled}
                      onChange={(event) => toggleRunSweep(event.target.checked)}
                    />
                    Run Sweep
                    <FieldHelp text="A sweep runs multiple training experiments with different parameter combinations so you can compare results and find better-performing settings." />
                  </label>
                  <div className="mt-2 flex items-center gap-3">
                    <button
                      type="button"
                      className="rounded-md border border-zinc-300 bg-white px-2.5 py-1 text-[11px] font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 disabled:cursor-not-allowed disabled:text-zinc-400 disabled:shadow-none"
                      onClick={reloadSweepValues}
                      disabled={!runSweepEnabled || isTraining}
                    >
                      Reload
                    </button>
                    <span className="text-[11px] text-zinc-500">Toggle ON to use sweep values. Use Reload for a fresh random sweep.</span>
                  </div>
                </div>

                <div className="border-t border-zinc-100 pt-5">
                  <label className="inline-flex items-start gap-2 text-xs text-zinc-600">
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 accent-zinc-900"
                      checked={autoDistillEnabled}
                      onChange={(event) => setAutoDistillEnabled(event.target.checked)}
                      disabled={isTraining || isDistilling}
                    />
                    <span>
                      <span className="inline-flex items-center gap-1 font-semibold text-zinc-700">
                        Auto-distill Training Runs
                        <FieldHelp text="When enabled, each completed training run is distilled automatically after training completes. Use Show Distilled in the table to open metadata." />
                      </span>
                      <span className="mt-0.5 block text-zinc-500">
                        Smaller distilled models are created during training runs.
                      </span>
                    </span>
                  </label>
                  {distillStatus ? (
                    <p className="mt-1 text-xs text-zinc-500">{distillStatus}</p>
                  ) : null}
                </div>
              </div>
            </div>
            {trainingError ? (
              <p className="mt-3 text-xs text-red-600">{trainingError}</p>
            ) : null}
            <TrainingRunsSection
              trainingRuns={trainingRuns}
              copyRunsStatus={copyRunsStatus}
              isTraining={isTraining}
              isStopRequested={isStopRequested}
              onCopyTrainingRuns={onCopyTrainingRuns}
              onClearTrainingRuns={clearTrainingRuns}
              onStopTrainingRuns={onStopTrainingRuns}
              onDistillFromRun={onDistillFromRun}
              onSeeDistilledFromRun={onSeeDistilledFromRun}
              isDistillationSupportedForRun={isDistillationSupportedForRun}
              distillingTeacherKey={distillingTeacherKey}
              distilledByTeacher={distilledByTeacher}
            />
          </section>

          <details
            className="rounded-xl border border-zinc-200 bg-white p-4"
            open
          >
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Dataset Table Preview
            </summary>
            <p className="mt-3 text-xs text-zinc-500">
              Showing {rowCount} rows
              {totalRowCount > rowCount ? ` of ${totalRowCount}` : ""} for{" "}
              <code>{selectedDatasetId ?? "no selection"}</code>.
            </p>
            <div className="mt-3">
              <DataTable rows={tableRows} columns={tableColumns} height={360} maxWidth={980} />
            </div>
          </details>
        </div>
      </main>
      <OptimalParamsModal
        isOpen={isOptimalModalOpen}
        onClose={() => setIsOptimalModalOpen(false)}
        pendingOptimalParams={pendingOptimalParams}
        pendingOptimalPrediction={pendingOptimalPrediction}
        onApply={onApplyOptimalParams}
        activeAlgorithm={trainingMode}
      />
      <DistillMetricsModal
        isOpen={isDistillMetricsModalOpen}
        onClose={() => setIsDistillMetricsModalOpen(false)}
        distillMetrics={distillMetrics}
        distillModelId={distillModelId}
        distillModelPath={distillModelPath}
        distillComparison={distillComparison}
      />
      <ModelPreviewModal
        isOpen={isModelPreviewOpen}
        onClose={() => setIsModelPreviewOpen(false)}
        framework="pytorch"
        mode={trainingMode}
      />
    </div>
  );
}
