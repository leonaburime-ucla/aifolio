<script setup lang="ts">
import { toRef } from "vue";
import type { ChartSpec } from "~/composables/useChartStore";
import { useChartRendererOrchestrator } from "../orchestrator";

const props = defineProps<{ spec: ChartSpec }>();

const { chartEl } = useChartRendererOrchestrator(toRef(props, "spec"));
</script>

<template>
  <div class="relative rounded-xl border border-zinc-200 bg-white p-4">
    <div class="mb-4">
      <p v-if="spec.title" class="text-sm font-semibold text-zinc-900">
        {{ spec.title }}
      </p>
      <p v-if="spec.description" class="text-xs text-zinc-700/80 mt-1">
        {{ spec.description }}
      </p>
    </div>
    <div ref="chartEl" class="h-[300px] w-full" />
    <div v-if="spec.meta?.datasetLabel || spec.meta?.queryTimeMs != null" class="mt-3 text-xs text-zinc-700/80">
      <p v-if="spec.meta?.datasetLabel">Dataset: {{ spec.meta.datasetLabel }}</p>
      <p v-if="spec.meta?.queryTimeMs != null">Query time: {{ (spec.meta.queryTimeMs / 1000).toFixed(2) }}s</p>
    </div>
  </div>
</template>
