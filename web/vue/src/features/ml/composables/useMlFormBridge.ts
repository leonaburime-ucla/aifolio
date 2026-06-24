import { onMounted, onUnmounted, type Ref } from "vue";
import {
  applyPytorchBridgePatch,
  applyTensorflowBridgePatch,
  toBridgeCsv,
} from "@aifolio/frontend-core/ml-training";

type FormBridgeBindings = {
  selectedDatasetId: Ref<string | null>;
  trainingMode: Ref<string>;
  targetColumn: Ref<string>;
  task: Ref<string>;
  sweepEnabled: Ref<boolean>;
  epochValues: Ref<string>;
  batchSizes: Ref<string>;
  learningRates: Ref<string>;
  testSizes: Ref<string>;
  hiddenDims: Ref<string>;
  numHiddenLayers: Ref<string>;
  dropouts: Ref<string>;
  excludeColumns: Ref<string>;
  dateColumns: Ref<string>;
  autoDistill: Ref<boolean>;
  onTrain: () => Promise<void>;
  onDatasetChange: (id: string | null) => void;
};

type FormBridge = {
  applyPatch: (patch: Record<string, unknown>) => { applied: string[]; skipped: string[] };
  startTrainingRuns: () => Promise<{ status: string; reason?: string }>;
};

declare global {
  interface Window {
    __AIFOLIO_PYTORCH_FORM_BRIDGE__?: FormBridge;
    __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: FormBridge;
  }
}

function createBridge(bindings: FormBridgeBindings): FormBridge {
  return {
    applyPatch(patch) {
      const applied: string[] = [];
      const handled = new Set<string>();

      if (patch.dataset_id !== undefined) {
        bindings.onDatasetChange(String(patch.dataset_id));
        applied.push("dataset_id");
        handled.add("dataset_id");
      }
      if (patch.training_mode !== undefined) {
        bindings.trainingMode.value = String(patch.training_mode);
        applied.push("training_mode");
        handled.add("training_mode");
      }
      if (patch.target_column !== undefined) {
        bindings.targetColumn.value = String(patch.target_column);
        applied.push("target_column");
        handled.add("target_column");
      }
      if (patch.task !== undefined) {
        bindings.task.value = String(patch.task);
        applied.push("task");
        handled.add("task");
      }
      const runSweepValue = patch.run_sweep ?? patch.set_sweep_values;
      if (runSweepValue !== undefined) {
        bindings.sweepEnabled.value = Boolean(runSweepValue);
        applied.push("run_sweep");
        handled.add("run_sweep");
      }
      if (patch.epoch_values !== undefined) {
        bindings.epochValues.value = toBridgeCsv(patch.epoch_values);
        applied.push("epoch_values");
        handled.add("epoch_values");
      }
      if (patch.batch_sizes !== undefined) {
        bindings.batchSizes.value = toBridgeCsv(patch.batch_sizes);
        applied.push("batch_sizes");
        handled.add("batch_sizes");
      }
      if (patch.learning_rates !== undefined) {
        bindings.learningRates.value = toBridgeCsv(patch.learning_rates);
        applied.push("learning_rates");
        handled.add("learning_rates");
      }
      if (patch.test_sizes !== undefined) {
        bindings.testSizes.value = toBridgeCsv(patch.test_sizes);
        applied.push("test_sizes");
        handled.add("test_sizes");
      }
      if (patch.hidden_dims !== undefined) {
        bindings.hiddenDims.value = toBridgeCsv(patch.hidden_dims);
        applied.push("hidden_dims");
        handled.add("hidden_dims");
      }
      if (patch.num_hidden_layers !== undefined) {
        bindings.numHiddenLayers.value = toBridgeCsv(patch.num_hidden_layers);
        applied.push("num_hidden_layers");
        handled.add("num_hidden_layers");
      }
      if (patch.dropouts !== undefined) {
        bindings.dropouts.value = toBridgeCsv(patch.dropouts);
        applied.push("dropouts");
        handled.add("dropouts");
      }
      if (patch.exclude_columns !== undefined) {
        bindings.excludeColumns.value = toBridgeCsv(patch.exclude_columns);
        applied.push("exclude_columns");
        handled.add("exclude_columns");
      }
      if (patch.date_columns !== undefined) {
        bindings.dateColumns.value = toBridgeCsv(patch.date_columns);
        applied.push("date_columns");
        handled.add("date_columns");
      }
      if (patch.auto_distill !== undefined) {
        bindings.autoDistill.value = Boolean(patch.auto_distill);
        applied.push("auto_distill");
        handled.add("auto_distill");
      }

      const skipped = Object.keys(patch).filter((key) => !handled.has(key));
      return { applied, skipped };
    },

    async startTrainingRuns() {
      try {
        await new Promise<void>((r) => setTimeout(r, 0));
        await bindings.onTrain();
        return { status: "ok" };
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown training bridge error.";
        return { status: "error", reason };
      }
    },
  };
}

export function usePytorchFormBridge(bindings: FormBridgeBindings): void {
  onMounted(() => {
    if (typeof window === "undefined") return;
    window.__AIFOLIO_PYTORCH_FORM_BRIDGE__ = createBridge(bindings);
  });

  onUnmounted(() => {
    if (typeof window !== "undefined") {
      delete window.__AIFOLIO_PYTORCH_FORM_BRIDGE__;
    }
  });
}

export function useTensorflowFormBridge(bindings: FormBridgeBindings): void {
  onMounted(() => {
    if (typeof window === "undefined") return;
    window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = createBridge(bindings);
  });

  onUnmounted(() => {
    if (typeof window !== "undefined") {
      delete window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
    }
  });
}
