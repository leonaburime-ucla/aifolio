<script setup lang="ts">
import { useAgenticResearchOrchestrator } from "~/features/agentic-research/orchestrator";
import DatasetCombobox from "~/components/General/DatasetCombobox.vue";
import DataTable from "~/components/Datatable/DataTable.vue";
import ChartRenderer from "~/features/recharts/components/ChartRenderer.vue";

const emit = defineEmits<{ "dataset-change": [id: string] }>();


const {
  datasetOptions,
  selectedDatasetId,
  tableRows,
  tableColumns,
  sklearnTools,
  chartSpecs,
  isLoading,
  error,
  toolGroups,
  samplePrompts,
  onDatasetChange,
  removeChartSpec,
} = useAgenticResearchOrchestrator({
  baseUrl: "/api/ai",
  onDatasetChange: (id) => emit("dataset-change", id),
});
</script>

<template>
  <div class="flex flex-col gap-6">
    <!-- ML Algorithms + Sample Prompts -->
    <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
      <summary class="cursor-pointer text-[12px] font-semibold">
        ML Algorithms + Sample Prompts
      </summary>
      <div class="mt-3 text-[12px]">
        <p class="font-bold text-red-600">Results take 1-2min</p>
        <div class="mt-3">
          <p class="font-bold text-zinc-900">Sample Prompts</p>
          <ol class="mt-2 list-decimal space-y-1 pl-4">
            <li v-for="prompt in samplePrompts" :key="prompt">{{ prompt }}</li>
          </ol>
        </div>
        <div v-if="sklearnTools.length > 0" class="mt-4 flex flex-col gap-2">
          <div v-for="group in toolGroups" :key="group.name">
            <p class="text-[11px] font-semibold uppercase tracking-wide">{{ group.name }}</p>
            <p class="text-[12px]">{{ group.formatted }}</p>
          </div>
        </div>
        <p v-else class="mt-1">Loading...</p>
      </div>
    </details>

    <!-- Preprocessing Notes -->
    <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-[12px] text-zinc-600">
      <summary class="cursor-pointer font-semibold text-zinc-900">
        Preprocessing Notes
      </summary>
      <div class="mt-3 flex flex-col gap-2">
        <p><strong>Categorical Encoding:</strong> Text columns with &le; 20 unique values are automatically One-Hot Encoded.</p>
        <p><strong>High Cardinality &amp; IDs:</strong> Text columns with &gt; 20 unique values or ID-like names are dropped to prevent feature explosion.</p>
        <p><strong>Date Parsing:</strong> Dates and timestamps are extracted into Year, Month, and Day numeric features.</p>
        <p><strong>Missing Values:</strong> Missing numeric values are imputed using the column median to maintain robustness against outliers.</p>
        <p><strong>Feature Scaling:</strong> All features are standardized to zero mean and unit variance (StandardScaler) before analysis. This prevents large-range features from dominating algorithms like PCA.</p>
      </div>
    </details>

    <!-- Dataset -->
    <div class="mt-4">
      <div class="mb-4 flex flex-col gap-2">
        <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">Dataset</p>
        <DatasetCombobox
          :options="datasetOptions"
          :selected-id="selectedDatasetId"
          @change="onDatasetChange"
        />
      </div>

      <!-- Charts -->
      <details class="rounded-2xl border border-zinc-200 bg-white/60 p-4 shadow-sm backdrop-blur-sm" open>
        <summary class="cursor-pointer text-sm font-semibold text-zinc-900">Charts</summary>
        <div class="mt-4">
          <div v-if="isLoading" class="h-56 animate-pulse rounded-xl bg-zinc-100" />
          <div v-else-if="chartSpecs.length > 0" :class="['flex flex-col gap-4', chartSpecs.length > 2 ? 'max-h-[56rem] overflow-y-auto pr-2' : '']">
            <ChartRenderer
              v-for="spec in chartSpecs"
              :key="spec.id"
              :spec="spec"
              removable
              @remove="removeChartSpec"
            />
          </div>
          <div v-else class="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-6 text-sm text-zinc-500">
            {{ error ?? 'No analysis chart data available yet.' }}
          </div>
        </div>
      </details>
    </div>

    <!-- Data Table -->
    <DataTable :rows="tableRows" :columns="tableColumns" />
  </div>
</template>
