<script setup lang="ts">
import { computed } from "vue";
import { getTensorflowModeExplainer } from "@aifolio/frontend-core/ml-training";
import DatasetCombobox from "~/components/General/DatasetCombobox.vue";
import DataTable from "~/components/Datatable/DataTable.vue";
import TrainingRunsTable from "~/features/ml/components/TrainingRunsTable.vue";
import ModelPreviewModal from "~/features/ml/components/ModelPreviewModal.vue";
import OptimalParamsModal from "~/features/ml/components/OptimalParamsModal.vue";
import DistillMetricsModal from "~/features/ml/components/DistillMetricsModal.vue";
import { useTrainingScreenOrchestrator } from "~/features/ml/orchestrator";

const {
  datasetOptions, selectedDatasetId, tableRows, tableColumns, targetColumn, datasetError,
  onDatasetChange,
  isTraining, stopTraining, trainingError, trainingRuns, trainingProgress,
  trainingMode, task, epochValues, batchSizes, learningRates, testSizes,
  hiddenDims, numHiddenLayers, dropouts, excludeColumns, dateColumns,
  sweepEnabled, autoDistill,
  isLinearBaseline, plannedRunCount, isTrainDisabled,
  onTrain, onCopyResults,
  isModelPreviewOpen, isOptimalModalOpen, pendingOptimalParams, pendingOptimalPrediction,
  optimizerStatus, isDistillMetricsModalOpen, distillMetrics, distillModelId,
  distillModelPath, distillComparison, distillingTeacherKey, distilledByTeacher,
  distillStatus, copyRunsStatus, isStopRequested, toggleRunSweep, reloadSweepValues,
  onFindOptimalParamsClick, onApplyOptimalParams, onDistillFromRun, onSeeDistilledFromRun,
  epochsValidation, testSizesValidation, learningRatesValidation, batchSizesValidation,
  hiddenDimsValidation, numHiddenLayersValidation, dropoutsValidation, defaults, completedRuns
} = useTrainingScreenOrchestrator({
  baseUrl: "/api/ai",
  framework: "tensorflow",
  defaultTrainingMode: "wide_and_deep",
  defaultExcludeColumns: "customerID",
});

const modeExplainer = computed(() => getTensorflowModeExplainer(trainingMode.value));
</script>

<template>
  <div class="flex min-h-screen flex-row bg-white text-zinc-900">
    <main class="min-w-0 flex-1 py-10">
      <div class="mx-auto flex max-w-5xl flex-col gap-4 px-6">
        <p class="text-sm font-semibold uppercase tracking-widest text-zinc-500">
          Machine Learning with TensorFlow
        </p>

        <!-- Dataset -->
        <div class="mt-2 flex max-w-xl flex-col gap-2">
          <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Dataset (CSV/XLS/XLSX)</p>
          <DatasetCombobox :options="datasetOptions" :selected-id="selectedDatasetId" @change="onDatasetChange" />
          <p v-if="datasetError" class="text-xs text-red-600">{{ datasetError }}</p>
        </div>

        <!-- Training Algorithm -->
        <section class="rounded-xl border border-zinc-200 bg-white p-4">
          <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Training Algorithm</p>
          <div class="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Select the machine learning architecture to run for this dataset.</span>
              <select v-model="trainingMode" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900">
                <option value="wide_and_deep">wide &amp; deep</option>
                <option value="entity_embeddings">entity embeddings</option>
                <option value="autoencoder_head">autoencoder + head</option>
                <option value="quantile_regression">quantile regression (p80)</option>
                <option value="multi_task_learning">multi-task learning</option>
                <option value="time_aware_tabular">time-aware tabular</option>
              </select>
              <div class="mt-2">
                <button
                  type="button"
                  class="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white hover:bg-zinc-800 transition"
                  @click="isModelPreviewOpen = true"
                >
                  Show Model
                </button>
              </div>
            </label>
            <div class="md:col-span-2 rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-900">
              <p><span class="font-semibold">What it is:</span> {{ modeExplainer.what }}</p>
              <p class="mt-1"><span class="font-semibold">Why it's unique:</span> {{ modeExplainer.why }}</p>
              <p class="mt-1"><span class="font-semibold">Distillation Note:</span> {{ modeExplainer.distillationNote }}</p>
            </div>
          </div>
        </section>

        <!-- Hyperparameters -->
        <section class="rounded-xl border border-zinc-200 bg-white p-4">
          <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Train TensorFlow Model</p>
          <div class="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Target Column</span>
              <select v-model="targetColumn" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900">
                <option value="">{{ defaults?.targetColumn || 'Select target column' }}</option>
                <option v-for="col in tableColumns" :key="col" :value="col">{{ col }}</option>
              </select>
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Task</span>
              <select v-model="task" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900">
                <option value="auto">auto</option>
                <option value="classification">classification</option>
                <option value="regression">regression</option>
              </select>
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Epochs</span>
              <input v-model="epochValues" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. 10,20,50" />
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Batch Sizes</span>
              <input v-model="batchSizes" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. 32,64" />
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Learning Rates</span>
              <input v-model="learningRates" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. 0.001" />
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Test Sizes</span>
              <input v-model="testSizes" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. 0.2" />
            </label>
            <label :class="['flex flex-col gap-1 text-xs', isLinearBaseline ? 'text-zinc-400' : 'text-zinc-600']">
              <span>Hidden Dims</span>
              <input v-model="hiddenDims" :disabled="isLinearBaseline" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100" placeholder="e.g. 128,256" />
            </label>
            <label :class="['flex flex-col gap-1 text-xs', isLinearBaseline ? 'text-zinc-400' : 'text-zinc-600']">
              <span>Hidden Layers</span>
              <input v-model="numHiddenLayers" :disabled="isLinearBaseline" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100" placeholder="e.g. 2,3,4" />
            </label>
            <label :class="['flex flex-col gap-1 text-xs', isLinearBaseline ? 'text-zinc-400' : 'text-zinc-600']">
              <span>Dropouts</span>
              <input v-model="dropouts" :disabled="isLinearBaseline" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100" placeholder="e.g. 0.1,0.2" />
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Exclude Columns</span>
              <input v-model="excludeColumns" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. customerID,Order,PID" />
              <span class="text-[10px] text-zinc-400">Preloaded: {{ defaults?.excludeColumns.length ? defaults.excludeColumns.join(', ') : '(none)' }}</span>
            </label>
            <label class="flex flex-col gap-1 text-xs text-zinc-600">
              <span>Date Columns</span>
              <input v-model="dateColumns" class="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900" placeholder="e.g. Date" />
              <span class="text-[10px] text-zinc-400">Preloaded: {{ defaults?.dateColumns.length ? defaults.dateColumns.join(', ') : '(none)' }}</span>
            </label>
          </div>

          <!-- Hyperparameter Validation Warnings -->
          <div class="mt-2 grid max-w-3xl grid-cols-1 gap-1 text-xs text-zinc-500 md:grid-cols-2">
            <p :class="epochsValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Epochs: {{ epochsValidation.ok ? epochsValidation.values.join(', ') : epochsValidation.error }}
            </p>
            <p :class="batchSizesValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Batch sizes: {{ batchSizesValidation.ok ? batchSizesValidation.values.join(', ') : batchSizesValidation.error }}
            </p>
            <p :class="learningRatesValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Learning rates: {{ learningRatesValidation.ok ? learningRatesValidation.values.join(', ') : learningRatesValidation.error }}
            </p>
            <p :class="testSizesValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Test sizes: {{ testSizesValidation.ok ? testSizesValidation.values.join(', ') : testSizesValidation.error }}
            </p>
            <p :class="isLinearBaseline || hiddenDimsValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Hidden dims: {{ isLinearBaseline ? 'n/a (linear baseline)' : (hiddenDimsValidation.ok ? hiddenDimsValidation.values.join(', ') : hiddenDimsValidation.error) }}
            </p>
            <p :class="isLinearBaseline || numHiddenLayersValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Hidden layers: {{ isLinearBaseline ? 'n/a (linear baseline)' : (numHiddenLayersValidation.ok ? numHiddenLayersValidation.values.join(', ') : numHiddenLayersValidation.error) }}
            </p>
            <p :class="isLinearBaseline || dropoutsValidation.ok ? 'text-zinc-500' : 'text-red-600'">
              Dropouts: {{ isLinearBaseline ? 'n/a (linear baseline)' : (dropoutsValidation.ok ? dropoutsValidation.values.join(', ') : dropoutsValidation.error) }}
            </p>
          </div>

          <!-- Train Button -->
          <div class="mt-5 flex items-center gap-4">
            <button
              type="button"
              class="rounded-md bg-zinc-900 px-6 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-400 disabled:shadow-none transition"
              :disabled="isTrainDisabled"
              @click="onTrain"
            >
              {{ isTraining ? `Training ${trainingProgress.current}/${trainingProgress.total}...` : 'Train Model' }}
            </button>
            <div class="flex flex-col gap-0.5">
              <p class="text-xs text-zinc-500">Dataset: <code>{{ selectedDatasetId ?? 'none' }}</code></p>
              <p class="text-xs font-semibold text-red-600">Planned runs: {{ plannedRunCount }}</p>
            </div>
          </div>
          <p v-if="trainingError" class="mt-3 text-xs text-red-600">{{ trainingError }}</p>

          <!-- Optional Settings -->
          <div class="mt-6 border-t border-zinc-200 pt-5">
            <p class="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-4 text-center">Optional Settings</p>
            <div class="grid gap-6 md:grid-cols-2">
              <div class="rounded-md border border-zinc-200 p-4 text-xs space-y-1 bg-zinc-50/20">
                <p class="font-semibold text-zinc-900 text-sm">Bayesian Optimization</p>
                <p class="text-zinc-600">What is it: A method for optimizing expensive black-box functions by using a probabilistic model to choose promising parameter settings.</p>
                <p class="text-zinc-600">How it works: Uses completed runs to suggest the next promising hyperparameter combination. <span class="font-semibold text-blue-600">Requires at least 5 completed runs.</span></p>
                <div class="pt-2 flex items-center gap-2">
                  <button
                    type="button"
                    class="rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 disabled:cursor-not-allowed disabled:text-zinc-400 transition"
                    :disabled="isTraining || completedRuns.length < 5"
                    @click="onFindOptimalParamsClick"
                  >
                    Find Optimal Params
                  </button>
                  <span v-if="optimizerStatus" class="text-[11px] text-zinc-500 font-medium">{{ optimizerStatus }}</span>
                </div>
              </div>

              <div class="flex flex-col gap-4 rounded-md border border-zinc-200 p-4 text-xs bg-zinc-50/20">
                <div class="space-y-2">
                  <label class="flex items-center gap-2 cursor-pointer select-none">
                    <input type="checkbox" :checked="sweepEnabled" @change="toggleRunSweep(!sweepEnabled)" :disabled="isTraining" class="rounded border-zinc-300 accent-zinc-900 h-4 w-4" />
                    <span class="font-medium text-zinc-700">Set Sweep Values</span>
                  </label>
                  <div class="flex items-center gap-2">
                    <button
                      type="button"
                      :disabled="!sweepEnabled || isTraining"
                      @click="reloadSweepValues"
                      class="rounded-md border border-zinc-300 bg-white px-2.5 py-1 text-[11px] font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 disabled:cursor-not-allowed disabled:text-zinc-400 transition"
                    >
                      Reload
                    </button>
                    <span class="text-[10px] text-zinc-400">Toggle ON to apply sweep values. Use Reload for fresh sweep values.</span>
                  </div>
                </div>

                <div class="border-t border-zinc-200 pt-3">
                  <label class="flex items-start gap-2 cursor-pointer select-none">
                    <input type="checkbox" v-model="autoDistill" :disabled="isTraining" class="rounded border-zinc-300 accent-zinc-900 h-4 w-4 mt-0.5" />
                    <span class="flex flex-col">
                      <span class="font-medium text-zinc-700">Auto-distill Training Runs</span>
                      <span class="text-[10px] text-zinc-400">Smaller distilled models are created automatically during training sweeps.</span>
                    </span>
                  </label>
                  <p v-if="distillStatus" class="text-xs text-zinc-500 mt-1 font-medium">{{ distillStatus }}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Training Runs -->
        <section class="mt-4 border-t border-zinc-200 pt-4">
          <div class="mb-2 flex items-center justify-between">
            <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Training Runs</p>
            <div class="flex items-center gap-2">
              <span v-if="copyRunsStatus" class="text-xs text-zinc-500">{{ copyRunsStatus }}</span>
              <button
                type="button"
                :disabled="trainingRuns.length === 0"
                class="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                @click="onCopyResults"
              >
                Copy Results
              </button>
              <button
                type="button"
                :disabled="trainingRuns.length === 0"
                class="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                @click="trainingRuns = []"
              >
                Clear Runs
              </button>
              <button
                type="button"
                :disabled="!isTraining || isStopRequested"
                class="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:bg-red-300"
                :aria-busy="isStopRequested"
                @click="stopTraining = true"
              >
                {{ isStopRequested ? 'Stop Requested...' : 'Stop Training Runs' }}
              </button>
            </div>
          </div>
          <p v-if="isStopRequested" class="mb-2 text-xs text-amber-700">
            Stop requested. Current run will finish, then remaining runs are canceled.
          </p>
          <div v-if="trainingRuns.length === 0" class="text-xs text-zinc-500">
            No runs yet. Train once to populate the results table.
          </div>
          <TrainingRunsTable
            v-else
            :runs="trainingRuns"
            framework="tensorflow"
            :distilling-teacher-key="distillingTeacherKey"
            :distilled-by-teacher="distilledByTeacher"
            @distill="onDistillFromRun"
            @see-distilled="onSeeDistilledFromRun"
          />
        </section>

        <!-- Preprocessing Notes -->
        <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-[12px] text-zinc-600">
          <summary class="cursor-pointer font-semibold text-zinc-900">Preprocessing Notes</summary>
          <div class="mt-3 flex flex-col gap-2">
            <p><strong>Categorical Encoding:</strong> Text columns with &le; 20 unique values are automatically One-Hot Encoded.</p>
            <p><strong>High Cardinality &amp; IDs:</strong> Text columns with &gt; 20 unique values or ID-like names are dropped to prevent feature explosion.</p>
            <p><strong>Date Parsing:</strong> Dates and timestamps are extracted into Year, Month, and Day numeric features.</p>
            <p><strong>Missing Values:</strong> Missing numeric values are imputed using the column median to maintain robustness against outliers.</p>
            <p><strong>Feature Scaling:</strong> All features are standardized to zero mean and unit variance (StandardScaler) before analysis.</p>
          </div>
        </details>

        <!-- Data Table Preview -->
        <details class="rounded-xl border border-zinc-200 bg-white p-4" open>
          <summary class="cursor-pointer text-xs font-semibold uppercase tracking-wide text-zinc-500">Dataset Table Preview</summary>
          <p class="mt-3 text-xs text-zinc-500">
            Showing {{ tableRows.length }} rows for <code>{{ selectedDatasetId ?? 'no selection' }}</code>.
          </p>
          <div class="mt-3">
            <DataTable :rows="tableRows" :columns="tableColumns" />
          </div>
        </details>
      </div>
    </main>

    <!-- Modals parity -->
    <ModelPreviewModal
      :is-open="isModelPreviewOpen"
      framework="tensorflow"
      :mode="trainingMode"
      @close="isModelPreviewOpen = false"
    />

    <OptimalParamsModal
      :is-open="isOptimalModalOpen"
      :pending-optimal-params="pendingOptimalParams"
      :pending-optimal-prediction="pendingOptimalPrediction"
      :active-algorithm="trainingMode"
      @close="isOptimalModalOpen = false"
      @apply="onApplyOptimalParams"
    />

    <DistillMetricsModal
      :is-open="isDistillMetricsModalOpen"
      :distill-metrics="distillMetrics"
      :distill-model-id="distillModelId"
      :distill-model-path="distillModelPath"
      :distill-comparison="distillComparison"
      @close="isDistillMetricsModalOpen = false"
    />
  </div>
</template>
