<template>
  <div class="mt-4 overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
    <div
      class="overflow-auto"
      :style="{ height: `${trainingTableHeight}px`, maxWidth: '980px' }"
    >
      <table class="min-w-[900px] w-full text-left text-sm">
        <thead class="sticky top-0 z-10 bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
          <tr>
            <th
              v-for="column in columns"
              :key="column"
              class="px-4 py-3 font-semibold"
            >
              <button
                type="button"
                class="inline-flex items-center gap-1 text-left"
                @click="toggleSort(column)"
              >
                <span>{{ column }}</span>
                <span v-if="sortKey === column">{{ sortDirection === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-zinc-200">
          <tr
            v-for="(row, index) in sortedRows"
            :key="getRowKey(row, index)"
            class="hover:bg-zinc-50"
            style="height: 48px"
          >
            <td
              v-for="column in columns"
              :key="column"
              class="px-4 py-3 text-zinc-700"
            >
              <template v-if="column === 'distill_action'">
                <span
                  v-if="getAction(row).kind === 'student_model'"
                  class="inline-flex rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700"
                >
                  Student Model
                </span>
                <span
                  v-else-if="getAction(row).kind === 'not_available'"
                  class="text-xs text-zinc-400"
                >
                  Not Available
                </span>
                <button
                  v-else-if="getAction(row).kind === 'show_distilled'"
                  type="button"
                  class="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700"
                  @click="$emit('see-distilled', row)"
                >
                  Show Distilled
                </button>
                <button
                  v-else
                  type="button"
                  class="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                  :disabled="getAction(row).isDistillingThisRow"
                  @click="$emit('distill', row)"
                >
                  {{ getAction(row).isDistillingThisRow ? "Distilling..." : "Distill" }}
                </button>
              </template>
              <template v-else>
                {{ formatCellValue(row[column]) }}
              </template>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import type { TrainingRunRow } from "@aifolio/contracts/entities/ml-training";
import {
  TRAINING_RUN_COLUMNS,
  buildDistillActionModel,
  calcTrainingTableHeight,
} from "@aifolio/frontend-core/ml-training";

const props = withDefaults(
  defineProps<{
    runs: TrainingRunRow[];
    framework: "pytorch" | "tensorflow";
    distillingTeacherKey?: string | null;
    distilledByTeacher?: Record<string, string>;
  }>(),
  {
    distillingTeacherKey: null,
    distilledByTeacher: () => ({}),
  }
);

defineEmits<{
  distill: [row: TrainingRunRow];
  "see-distilled": [row: TrainingRunRow];
}>();

const columns = [...TRAINING_RUN_COLUMNS];
const sortKey = ref<string | null>(null);
const sortDirection = ref<"asc" | "desc">("asc");

const trainingTableHeight = computed(() =>
  calcTrainingTableHeight({ rowsCount: props.runs.length })
);

const sortedRows = computed(() => {
  if (!sortKey.value) return props.runs;

  return [...props.runs].sort((a, b) => {
    const direction = sortDirection.value === "asc" ? 1 : -1;
    return compareCellValues(a[sortKey.value!], b[sortKey.value!]) * direction;
  });
});

function toggleSort(column: string) {
  if (sortKey.value === column) {
    sortDirection.value = sortDirection.value === "asc" ? "desc" : "asc";
    return;
  }
  sortKey.value = column;
  sortDirection.value = "asc";
}

function compareCellValues(
  left: TrainingRunRow[string] | undefined,
  right: TrainingRunRow[string] | undefined
) {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;

  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
    return leftNumber - rightNumber;
  }

  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function formatCellValue(value: TrainingRunRow[string] | undefined) {
  if (value == null || value === "") return "";
  return String(value);
}

function getRowKey(row: TrainingRunRow, index: number) {
  return String(row.run_id ?? row.model_id ?? row.completed_at ?? index);
}

function isDistillationSupportedForRun(row: TrainingRunRow) {
  const mode = String(row.training_mode ?? "");
  if (props.framework === "tensorflow") {
    return ["mlp_dense", "linear_glm_baseline", "wide_and_deep"].includes(mode);
  }
  return ["mlp_dense", "linear_glm_baseline", "tabresnet"].includes(mode);
}

function getAction(row: TrainingRunRow) {
  return buildDistillActionModel({
    row,
    isDistillationSupportedForRun,
    distillingTeacherKey: props.distillingTeacherKey,
    distilledByTeacher: props.distilledByTeacher,
  });
}
</script>
