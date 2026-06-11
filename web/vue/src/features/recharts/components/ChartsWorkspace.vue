<script setup lang="ts">
import { useChartsWorkspaceOrchestrator } from "../orchestrator";
import ChartRenderer from "./ChartRenderer.vue";

const { chartSpecs, removeChartSpec } = useChartsWorkspaceOrchestrator();
</script>

<template>
  <div class="flex flex-col gap-8">
    <!-- How to Use Page + Prompts to Try -->
    <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
      <summary class="cursor-pointer text-[12px] font-semibold">
        How to Use Page + Prompts to Try
      </summary>
      <div class="mt-3 text-[12px] text-zinc-700">
        <p>This AI Chat creates charts(Recharts and Echarts) from internal sample data. I do not have APIs for real-time data. All data is from the LLM's internal models.</p>
        <div class="mt-3">
          <p class="font-bold text-zinc-900">Sample prompts to try:</p>
          <ul class="mt-2 list-disc space-y-1 pl-4">
            <li>Create a line chart of Solana and Bitcoin for the past 5 months.</li>
            <li>Create an area chart of Peruvian beef exports over the past 15 years.</li>
            <li>Show a line chart of Manhattan vs London vs Paris average rent since 2000 as a share of average salary in each of those cities respectively.</li>
            <li>Plot a line chart of US debt levels for the past 50 years. Estimate what it will be for the next 20 in a blue line</li>
            <li>Make a scatter chart comparing Bitcoin and Ethereum returns over the last 30 days.</li>
          </ul>
        </div>
      </div>
    </details>

    <div
      v-if="chartSpecs.length === 0"
      class="rounded-2xl border border-dashed border-zinc-300 bg-white p-6 text-sm text-zinc-500"
    >
      Charts generated from chat will appear here.
    </div>
    <div v-else class="flex flex-col gap-6">
      <div v-for="spec in chartSpecs" :key="spec.id" class="relative">
        <button
          type="button"
          class="absolute -right-2 -top-2 z-10 flex h-7 w-7 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-500 shadow-sm transition hover:bg-zinc-50"
          aria-label="Remove chart"
          @click="removeChartSpec(spec.id)"
        >
          &times;
        </button>
        <ChartRenderer :spec="spec" />
      </div>
    </div>
  </div>
</template>
