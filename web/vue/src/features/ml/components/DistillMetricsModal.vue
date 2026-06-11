<template>
  <Modal :is-open="isOpen" @close="$emit('close')" title="Distillation Metrics">
    <div class="space-y-3 p-1 text-sm text-zinc-700">
      <div class="grid grid-cols-2 gap-2 md:grid-cols-4">
        <div class="rounded-md border border-zinc-200 bg-zinc-50 p-2">
          <p class="text-[11px] uppercase tracking-wide text-zinc-500 font-semibold">metric_name</p>
          <p class="mt-1 font-semibold text-zinc-900">{{ distillMetrics?.test_metric_name ?? 'n/a' }}</p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-zinc-50 p-2">
          <p class="text-[11px] uppercase tracking-wide text-zinc-500 font-semibold">metric_score</p>
          <p class="mt-1 font-semibold text-zinc-900">
            {{ formatMetricNumber({ value: distillMetrics?.test_metric_value }) }}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-zinc-50 p-2">
          <p class="text-[11px] uppercase tracking-wide text-zinc-500 font-semibold">train_loss</p>
          <p class="mt-1 font-semibold text-zinc-900">{{ formatMetricNumber({ value: distillMetrics?.train_loss }) }}</p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-zinc-50 p-2">
          <p class="text-[11px] uppercase tracking-wide text-zinc-500 font-semibold">test_loss</p>
          <p class="mt-1 font-semibold text-zinc-900">{{ formatMetricNumber({ value: distillMetrics?.test_loss }) }}</p>
        </div>
      </div>

      <div v-if="distillComparison" class="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 space-y-2">
        <p class="font-semibold text-zinc-800 text-sm">Teacher vs Student</p>
        <p class="mt-1">
          metric ({{ distillComparison.metricName }}): teacher
          <span class="font-medium text-zinc-900">
            {{ formatMetricNumber({ value: distillComparison.teacherMetricValue }) }}
          </span>
          | student
          <span class="font-medium text-zinc-900">
            {{ formatMetricNumber({ value: distillComparison.studentMetricValue }) }}
          </span>
        </p>
        <p class="mt-1">
          quality delta (student vs teacher):
          <span class="font-medium text-zinc-900">
            {{ formatMetricNumber({ value: distillComparison.qualityDelta }) }}
          </span>
          <span class="text-zinc-500">
            ({{ distillComparison.higherIsBetter ? 'higher is better' : 'lower is better' }})
          </span>
        </p>
        <p class="mt-1">
          model size: teacher
          <span class="font-medium text-zinc-900">
            {{ formatBytes({ value: distillComparison.teacherModelSizeBytes }) }}
          </span>
          | student
          <span class="font-medium text-zinc-900">
            {{ formatBytes({ value: distillComparison.studentModelSizeBytes }) }}
          </span>
        </p>
        <p class="mt-1">
          size saved:
          <span class="font-medium text-zinc-900">
            {{ formatBytes({ value: distillComparison.sizeSavedBytes }) }}
          </span>
          <span class="text-zinc-500">{{ sizeSavedLabel }}</span>
        </p>
        <p class="mt-1">
          params: teacher
          <span class="font-medium text-zinc-900">
            {{ formatInt({ value: distillComparison.teacherParamCount }) }}
          </span>
          | student
          <span class="font-medium text-zinc-900">
            {{ formatInt({ value: distillComparison.studentParamCount }) }}
          </span>
        </p>
        <p class="mt-1">
          params saved:
          <span class="font-medium text-zinc-900">
            {{ formatInt({ value: distillComparison.paramSavedCount }) }}
          </span>
          <span class="text-zinc-500">{{ paramsSavedLabel }}</span>
        </p>

        <div class="mt-2 rounded-md border border-zinc-200 bg-white p-2 text-zinc-600">
          <p class="font-semibold text-zinc-700">Parameter Math</p>
          <p class="mt-1">
            D = input feature columns: columns of the dataset. Categorical columns are expanded via one-hot encoding.
          </p>
          <p class="mt-1">
            H = hidden dim, L = hidden layers, C = output classes/targets.
          </p>
        </div>

        <p class="mt-2 break-words text-zinc-600">
          Teacher:
          (D={{ distillComparison.teacherInputDim ?? 'n/a' }}, H={{ distillComparison.teacherHiddenDim ?? 'n/a' }}, L={{ distillComparison.teacherNumHiddenLayers ?? 'n/a' }}, C={{ distillComparison.teacherOutputDim ?? 'n/a' }});
          total params = (D*H + H) + ((L-1)*(H*H + H)) + (H*C + C) + (2*H*L) =
          <span class="font-medium text-zinc-900">
            {{ formatInt({ value: distillComparison.teacherParamCount }) }}
          </span>
        </p>
        <p class="mt-1 break-words text-zinc-600">
          Student:
          (D={{ distillComparison.studentInputDim ?? 'n/a' }}, H={{ distillComparison.studentHiddenDim ?? 'n/a' }}, L={{ distillComparison.studentNumHiddenLayers ?? 'n/a' }}, C={{ distillComparison.studentOutputDim ?? 'n/a' }});
          total params = (D*H + H) + ((L-1)*(H*H + H)) + (H*C + C) + (2*H*L) =
          <span class="font-medium text-zinc-900">
            {{ formatInt({ value: distillComparison.studentParamCount }) }}
          </span>
        </p>
      </div>

      <div v-if="showModelArtifacts" class="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
        <p>
          model_id: <span class="font-semibold text-zinc-800">{{ distillModelId ?? 'n/a' }}</span>
        </p>
        <p class="mt-1 break-all">
          model_path: <span class="font-semibold text-zinc-800">{{ distillModelPath ?? 'n/a' }}</span>
        </p>
      </div>
      <p v-else class="text-xs text-zinc-500">
        Model files were not saved for this run.
      </p>

      <div class="flex justify-end pt-1">
        <button
          type="button"
          class="rounded-md bg-zinc-900 px-3 py-2 text-xs font-semibold text-white hover:bg-zinc-850 transition"
          @click="$emit('close')"
        >
          Close
        </button>
      </div>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from "vue";
import Modal from "~/components/General/Modal.vue";
import {
  formatBytes,
  formatInt,
  formatMetricNumber,
  formatPercentLabel,
  hasModelArtifacts,
} from "@aifolio/frontend-core/ml-training";

const props = defineProps<{
  isOpen: boolean;
  distillMetrics: any;
  distillModelId: string | null;
  distillModelPath: string | null;
  distillComparison: any;
}>();

defineEmits<{ close: [] }>();

const sizeSavedLabel = computed(() =>
  formatPercentLabel({
    value: props.distillComparison?.sizeSavedPercent,
    fallback: "(file-size savings unavailable when no artifact files are persisted)",
  })
);

const paramsSavedLabel = computed(() =>
  formatPercentLabel({
    value: props.distillComparison?.paramSavedPercent,
    fallback: "",
  })
);

const showModelArtifacts = computed(() =>
  hasModelArtifacts({
    modelId: props.distillModelId ?? undefined,
    modelPath: props.distillModelPath ?? undefined,
  })
);
</script>
