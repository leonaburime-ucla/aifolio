"use client";

import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import DataTable from "@/core/views/components/Datatable/DataTable";
import {
  usePytorchTrainingIntegration,
  type PytorchIntegrationArgs,
} from "@/features/ml/hooks/usePytorchTraining.hooks";
import { FieldHelp } from "@/features/ml/views/components/FieldHelp";
import { TrainingRunsSection } from "@/features/ml/views/components/TrainingRunsSection";
import {
  DistillMetricsModal,
  OptimalParamsModal,
} from "@/features/ml/views/components/MlTrainingModals";
import {
  runPytorchDistillation,
  runPytorchTraining,
} from "@/features/ml/orchestrators/pytorchTraining.orchestrator";

type PyTorchPageProps = {
  orchestrator?: (args?: PytorchIntegrationArgs) => ReturnType<typeof usePytorchTrainingIntegration>;
};

/** PyTorch training page wired through the Orc-BASH integration hook. */
export default function PyTorchPage({
  orchestrator = usePytorchTrainingIntegration,
}: PyTorchPageProps) {
  const {
    datasetOptions,
    selectedDatasetId,
    isLoading,
    error,
    tableRows,
    tableColumns,
    rowCount,
    totalRowCount,
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
    trainingRuns,
    copyRunsStatus,
    onCopyTrainingRuns,
    clearTrainingRuns,
    completedRuns,
    optimizerStatus,
    isOptimalModalOpen,
    setIsOptimalModalOpen,
    pendingOptimalParams,
    pendingOptimalPrediction,
    isDistillMetricsModalOpen,
    setIsDistillMetricsModalOpen,
    distillMetrics,
    distillModelId,
    distillModelPath,
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
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Hidden Dims
                  <FieldHelp text="Width of each hidden layer in the MLP. Larger values increase model capacity and cost. Range: 8-500." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={hiddenDimsInput}
                  onChange={(event) => setHiddenDimsInput(event.target.value)}
                  placeholder="e.g. 128,256"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Hidden Layers
                  <FieldHelp text="Number of hidden layers in the MLP. More layers can model complex patterns but may overfit. Range: 1-15." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={numHiddenLayersInput}
                  onChange={(event) => setNumHiddenLayersInput(event.target.value)}
                  placeholder="e.g. 2,3,4"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Dropouts
                  <FieldHelp text="Dropout probability per hidden layer (0 to 0.9). Helps regularization; too high can underfit." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={dropoutsInput}
                  onChange={(event) => setDropoutsInput(event.target.value)}
                  placeholder="e.g. 0.1,0.2"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Exclude Columns
                  <FieldHelp text="Columns to drop from training features (for example IDs) as 
                  they are simply noise.  Comma-separated list." />
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
              <p className={hiddenDimsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden dims: {hiddenDimsValidation.ok ? `${hiddenDimsValidation.values.join(", ")}` : hiddenDimsValidation.error}
              </p>
              <p className={numHiddenLayersValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden layers: {numHiddenLayersValidation.ok ? `${numHiddenLayersValidation.values.join(", ")}` : numHiddenLayersValidation.error}
              </p>
              <p className={dropoutsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Dropouts: {dropoutsValidation.ok ? `${dropoutsValidation.values.join(", ")}` : dropoutsValidation.error}
              </p>
            </div>
            <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                    onClick={onTrainClick}
                    disabled={isTraining || isDistilling || !selectedDatasetId || plannedRunCount === 0}
                  >
                    {isTraining
                      ? `Training ${trainingProgress.current}/${trainingProgress.total}...`
                      : "Train Model"}
                  </button>
                  <div className="flex flex-col gap-1">
                    <p className="text-xs text-zinc-500">
                      Dataset: <code>{selectedDatasetId ?? "none"}</code>
                    </p>
                    <p className="text-xs font-semibold text-red-600">
                      Planned runs: {plannedRunCount}
                    </p>
                  </div>
                </div>
                <div className="border-t border-zinc-200 pt-3">
                  <p className="mb-2 text-xs text-zinc-600">
                    <span className="font-semibold text-zinc-700">Bayesian Optimization</span>
                    : uses completed runs to suggest the next promising hyperparameter combination.
                  </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                    onClick={onFindOptimalParamsClick}
                    disabled={isTraining || isDistilling || completedRuns.length < 5}
                  >
                    Find Optimal Params
                  </button>
                  <FieldHelp text="Uses a Bayesian-style search over your previous runs to suggest the next hyperparameter set likely to improve model performance. Requires at least 5 completed runs." />
                  {optimizerStatus ? (
                    <span className="text-xs text-zinc-500">{optimizerStatus}</span>
                  ) : null}
                </div>
                </div>
                {/* Distill section intentionally hidden for now.
                <div className="mt-2 border-t border-zinc-200 pt-3">
                  <div className="mt-2 flex items-start gap-2">
                    <button
                      type="button"
                      className="min-w-[132px] whitespace-nowrap rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                      onClick={onDistillClick}
                      disabled={isTraining || isDistilling || completedRuns.length === 0}
                    >
                      {isDistilling ? "Distilling..." : "Distill Model"}
                    </button>
                    <p className="text-xs text-zinc-600">
                      <span className="font-semibold text-zinc-700">Knowledge Distillation:</span>{" "}
                      train a smaller student model to mimic a stronger teacher model while preserving
                      similar performance.
                    </p>
                  </div>
                  {distillStatus ? (
                    <p className="mt-1 text-xs text-zinc-500">{distillStatus}</p>
                  ) : null}
                  <label className="mt-2 inline-flex items-center gap-2 text-xs text-zinc-600">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 accent-zinc-900"
                      checked={saveDistilledModel}
                      onChange={(event) => setSaveDistilledModel(event.target.checked)}
                    />
                    Save distilled model to <code>ai/ml/artifacts</code>
                  </label>
                </div>
                */}
              </div>
              <div className="border-zinc-200 md:border-l md:pl-4">
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
                <div className="mt-2">
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                    onClick={reloadSweepValues}
                    disabled={!runSweepEnabled || isTraining}
                  >
                    Reload
                  </button>
                </div>
                <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-zinc-500">
                  <li>Toggle ON to use sweep values.</li>
                  <li>Toggle OFF restores your previous non-sweep values.</li>
                  <li>Use Reload to generate a fresh random sweep set.</li>
                </ul>
              </div>
            </div>
            {trainingError ? (
              <p className="mt-3 text-xs text-red-600">{trainingError}</p>
            ) : null}
            <TrainingRunsSection
              trainingRuns={trainingRuns}
              copyRunsStatus={copyRunsStatus}
              onCopyTrainingRuns={onCopyTrainingRuns}
              onClearTrainingRuns={clearTrainingRuns}
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
      {/* <div className="sticky top-0 h-screen w-[420px] shrink-0 overflow-hidden">
        <CopilotSidebar mode="ag-ui" />
      </div> */}
      <OptimalParamsModal
        isOpen={isOptimalModalOpen}
        onClose={() => setIsOptimalModalOpen(false)}
        pendingOptimalParams={pendingOptimalParams}
        pendingOptimalPrediction={pendingOptimalPrediction}
        onApply={onApplyOptimalParams}
      />
      <DistillMetricsModal
        isOpen={isDistillMetricsModalOpen}
        onClose={() => setIsDistillMetricsModalOpen(false)}
        distillMetrics={distillMetrics}
        distillModelId={distillModelId}
        distillModelPath={distillModelPath}
      />
    </div>
  );
}
