<template>
  <Modal :is-open="isOpen" @close="$emit('close')" title="Bayesian Optimization Suggestion">
    <div class="space-y-4 p-1 text-sm">
      <p class="text-zinc-600">
        Suggested next hyperparameters for
        <span v-if="activeAlgorithm" class="font-semibold text-zinc-800">{{ activeAlgorithm }}</span>
        <span v-else>this architecture</span>
        for better accuracy based on completed runs.
      </p>

      <div class="grid grid-cols-1 gap-2 text-zinc-800 md:grid-cols-2">
        <p>epochs: <span class="font-semibold">{{ pendingOptimalParams?.epochs ?? 'n/a' }}</span></p>
        <p>
          learning_rate:
          <span class="font-semibold">
            {{ pendingOptimalParams ? Number(pendingOptimalParams.learning_rate.toPrecision(6)) : 'n/a' }}
          </span>
        </p>
        <p>
          test_size:
          <span class="font-semibold">
            {{ pendingOptimalParams ? Number(pendingOptimalParams.test_size.toPrecision(4)) : 'n/a' }}
          </span>
        </p>
        <p>batch_size: <span class="font-semibold">{{ pendingOptimalParams?.batch_size ?? 'n/a' }}</span></p>
        <p>hidden_dim: <span class="font-semibold">{{ pendingOptimalParams?.hidden_dim ?? 'n/a' }}</span></p>
        <p>num_hidden_layers: <span class="font-semibold">{{ pendingOptimalParams?.num_hidden_layers ?? 'n/a' }}</span></p>
        <p>
          dropout:
          <span class="font-semibold">
            {{ pendingOptimalParams ? Number(pendingOptimalParams.dropout.toPrecision(4)) : 'n/a' }}
          </span>
        </p>
      </div>

      <p v-if="pendingOptimalPrediction" class="font-semibold text-red-600">
        Predicted: {{ pendingOptimalPrediction.metricName }} &approx;
        {{ formatMetricNumber({ value: pendingOptimalPrediction.metricValue }) }}
      </p>

      <div class="flex items-center justify-end gap-2 pt-2 border-t border-zinc-100">
        <button
          type="button"
          class="rounded-md border border-zinc-300 bg-white px-3 py-2 text-xs font-medium text-zinc-700 hover:bg-zinc-50 transition"
          @click="$emit('close')"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-md bg-zinc-900 px-3 py-2 text-xs font-medium text-white hover:bg-zinc-800 transition disabled:bg-zinc-400 disabled:cursor-not-allowed"
          :disabled="!pendingOptimalParams"
          @click="$emit('apply')"
        >
          Update Table With Values
        </button>
      </div>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import Modal from "~/components/General/Modal.vue";
import { formatMetricNumber } from "@aifolio/frontend-core/ml-training";

defineProps<{
  isOpen: boolean;
  pendingOptimalParams: any;
  pendingOptimalPrediction: any;
  activeAlgorithm: string;
}>();

defineEmits<{
  close: [];
  apply: [];
}>();
</script>
