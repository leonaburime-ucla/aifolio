<template>
  <slot />
</template>

<script setup lang="ts">
import { ref, onMounted, toRef } from "vue";
import { useFrontendTool, useAgentContext } from "@copilotkit/vue/v2";
import { z } from "zod";
import { useChartStore } from "~/composables/useChartStore";
import {
  handleSwitchAgUiTab,
  handleAddChartSpec,
  resolveMlFormPatchFromToolArgs,
  formatAddChartSpecToolResult,
  formatClearChartsToolResult,
  formatSetFormFieldsToolResult,
  formatChangeTargetColumnToolResult,
  formatRandomizeFormFieldsToolResult,
  formatStartTrainingRunsToolResult,
  formatTrainModelToolResult,
  resolveTargetColumnChangeFromOptions,
  buildRandomPytorchFormPatch,
  buildRandomTensorflowFormPatch,
  PYTORCH_FIELD_SELECTORS,
  TENSORFLOW_FIELD_SELECTORS,
} from "@aifolio/frontend-core/ag-ui";
import { handleAgenticSetActiveDataset, fetchAgenticDatasetManifest } from "@aifolio/frontend-core/agentic-research";
import { createMlTrainingApi } from "~/features/ml/api";

const props = defineProps<{
  activeTab: string;
  activeDatasetId: string | null;
  selectedModelId: string;
}>();

const emit = defineEmits<{
  switchTab: [tab: string];
  datasetChange: [id: string];
}>();

const chartStore = useChartStore();
const mlApi = createMlTrainingApi({ baseUrl: "/api/ai" });
const datasetManifest = ref<{ id: string; label: string }[]>([]);

onMounted(async () => {
  try {
    const entries = await fetchAgenticDatasetManifest({}, { runtimeDeps: { resolveBaseUrl: () => "/api/ai" } });
    datasetManifest.value = entries.map(e => ({ id: e.id, label: e.label }));
  } catch {}
});

// --- Bridge helpers ---

function getPytorchBridge() {
  if (typeof window === "undefined") return null;
  return (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__ ?? null;
}

function getTensorflowBridge() {
  if (typeof window === "undefined") return null;
  return (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ ?? null;
}

function getActiveMlFramework(): "pytorch" | "tensorflow" | null {
  if (props.activeTab === "pytorch" || props.activeTab === "tensorflow") return props.activeTab;
  return null;
}

function getSelectOptions(selector: string) {
  if (typeof document === "undefined") return [];
  const el = document.querySelector(selector);
  if (!(el instanceof HTMLSelectElement)) return [];
  const currentValue = el.value?.trim();
  return Array.from(el.options)
    .map((o) => o.value)
    .filter((v) => v && v.trim().length > 0)
    .map((v) => ({ value: v, isCurrent: v === currentValue }));
}

function getSelectValue(selector: string): string | null {
  if (typeof document === "undefined") return null;
  const el = document.querySelector(selector);
  if (!(el instanceof HTMLSelectElement)) return null;
  const v = el.value?.trim();
  return v || null;
}

async function ensureTab(tab: string): Promise<void> {
  if (props.activeTab !== tab) {
    emit("switchTab", tab);
    await new Promise<void>((r) => setTimeout(r, 600));
  }
}

// --- Chart Tools ---

useFrontendTool({
  name: "add_chart_spec",
  description: "Add one chart spec or an array of chart specs to the frontend chart store for immediate rendering.",
  parameters: z.object({ chartSpec: z.any().optional(), chartSpecs: z.array(z.any()).optional() }),
  handler: async (args) => {
    const result = handleAddChartSpec(
      { chartSpec: args.chartSpec, chartSpecs: args.chartSpecs },
      (spec: any) => chartStore.addChartSpec(spec),
    );
    return formatAddChartSpecToolResult(result);
  },
});

useFrontendTool({
  name: "clear_charts",
  description: "Clear all chart specs from the frontend chart store.",
  parameters: z.object({}),
  handler: async () => {
    chartStore.clearCharts();
    return formatClearChartsToolResult();
  },
});

// --- Navigation Tools ---

useFrontendTool({
  name: "switch_ag_ui_tab",
  description: "Switch the active workspace tab.",
  parameters: z.object({ tab: z.string() }),
  handler: async ({ tab }) => {
    const result = handleSwitchAgUiTab(tab);
    if (result.status === "ok") emit("switchTab", result.tab);
    return JSON.stringify(result);
  },
});

useFrontendTool({
  name: "set_active_dataset",
  description: "Switch the active dataset.",
  parameters: z.object({ dataset_id: z.string() }),
  handler: async ({ dataset_id }) => {
    if (!dataset_id) return JSON.stringify({ status: "error", code: "MISSING_DATASET_ID" });
    const result = handleAgenticSetActiveDataset(
      dataset_id,
      datasetManifest.value,
      (resolvedId) => emit("datasetChange", resolvedId),
    );
    return JSON.stringify(result);
  },
});

// --- Active ML Tools (generic, route to whichever tab is active) ---

useFrontendTool({
  name: "set_active_ml_form_fields",
  description: "Patch form fields on the currently active ML tab. Use this by default for generic ML prompts when the user does not explicitly name PyTorch or TensorFlow.",
  parameters: z.object({ fields: z.record(z.any()) }),
  handler: async (args) => {
    const framework = getActiveMlFramework();
    if (!framework) return "Unable to update ML form fields: ACTIVE_ML_TAB_REQUIRED.";
    const patch = resolveMlFormPatchFromToolArgs(args);
    await ensureTab(framework);
    const bridge = framework === "pytorch" ? getPytorchBridge() : getTensorflowBridge();
    if (!bridge) return `Unable to update ML form fields: ${framework.toUpperCase()}_FORM_UNAVAILABLE.`;
    const result = bridge.applyPatch(patch);
    return formatSetFormFieldsToolResult(framework === "pytorch" ? "PyTorch" : "TensorFlow", result);
  },
});

useFrontendTool({
  name: "change_active_ml_target_column",
  description: "Change the target column on the currently active ML tab.",
  parameters: z.object({
    target_column: z.string().optional(),
    mode: z.enum(["different", "random", "next"]).optional(),
  }),
  handler: async (args) => {
    const framework = getActiveMlFramework();
    if (!framework) return "Unable to change ML target column: ACTIVE_ML_TAB_REQUIRED.";
    await ensureTab(framework);
    const selector = framework === "pytorch" ? PYTORCH_FIELD_SELECTORS.target_column : TENSORFLOW_FIELD_SELECTORS.target_column;
    const nextTarget = resolveTargetColumnChangeFromOptions(getSelectOptions(selector), args);
    if (!nextTarget) return formatChangeTargetColumnToolResult(framework === "pytorch" ? "PyTorch" : "TensorFlow", undefined, { status: "error", code: "TARGET_COLUMN_UNAVAILABLE" });
    const bridge = framework === "pytorch" ? getPytorchBridge() : getTensorflowBridge();
    if (!bridge) return `Unable to change target column: ${framework.toUpperCase()}_FORM_UNAVAILABLE.`;
    const result = bridge.applyPatch({ target_column: nextTarget });
    return formatChangeTargetColumnToolResult(framework === "pytorch" ? "PyTorch" : "TensorFlow", nextTarget, result);
  },
});

useFrontendTool({
  name: "randomize_active_ml_form_fields",
  description: "Randomize form fields on the currently active ML tab.",
  parameters: z.object({
    confirm_randomize: z.boolean(),
    value_count: z.number().optional(),
    style: z.enum(["safe", "balanced", "aggressive"]).optional(),
    set_sweep_values: z.boolean().optional(),
    run_sweep: z.boolean().optional(),
    auto_distill: z.boolean().optional(),
    randomize_model_fields: z.boolean().optional(),
  }),
  handler: async (args) => {
    const framework = getActiveMlFramework();
    if (!framework) return "Unable to randomize ML form fields: ACTIVE_ML_TAB_REQUIRED.";
    if (!args.confirm_randomize) return JSON.stringify({ status: "error", code: "RANDOMIZE_CONFIRMATION_REQUIRED" });
    await ensureTab(framework);
    const patch = framework === "pytorch"
      ? buildRandomPytorchFormPatch(args, { getSelectValue: (f) => f === "pytorch_training_mode" ? getSelectValue(PYTORCH_FIELD_SELECTORS.training_mode) : f === "pytorch_task" ? getSelectValue(PYTORCH_FIELD_SELECTORS.task) : null })
      : buildRandomTensorflowFormPatch(args, { getSelectValue: (f) => f === "tensorflow_task" ? getSelectValue(TENSORFLOW_FIELD_SELECTORS.task) : null });
    const bridge = framework === "pytorch" ? getPytorchBridge() : getTensorflowBridge();
    if (!bridge) return `Unable to randomize: ${framework.toUpperCase()}_FORM_UNAVAILABLE.`;
    const result = bridge.applyPatch(patch);
    return formatRandomizeFormFieldsToolResult(framework === "pytorch" ? "PyTorch" : "TensorFlow", { status: "ok", randomized: true, patch, ...result });
  },
});

useFrontendTool({
  name: "start_active_ml_training_runs",
  description: "Start training runs on the currently active ML tab.",
  parameters: z.object({}),
  handler: async () => {
    const framework = getActiveMlFramework();
    if (!framework) return "Unable to start ML training runs: ACTIVE_ML_TAB_REQUIRED.";
    await ensureTab(framework);
    const bridge = framework === "pytorch" ? getPytorchBridge() : getTensorflowBridge();
    if (!bridge?.startTrainingRuns) return `Unable to start training: ${framework.toUpperCase()}_FORM_UNAVAILABLE.`;
    const result = await bridge.startTrainingRuns();
    return formatStartTrainingRunsToolResult(framework === "pytorch" ? "PyTorch" : "TensorFlow", result);
  },
});

// --- PyTorch-specific Tools ---

useFrontendTool({
  name: "set_pytorch_form_fields",
  description: "Set/patch PyTorch training form fields on the /ag-ui PyTorch tab.",
  parameters: z.object({ fields: z.record(z.any()) }),
  handler: async (args) => {
    const patch = resolveMlFormPatchFromToolArgs(args);
    await ensureTab("pytorch");
    const bridge = getPytorchBridge();
    if (!bridge) return "Unable to update PyTorch form fields: PYTORCH_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch(patch);
    return formatSetFormFieldsToolResult("PyTorch", result);
  },
});

useFrontendTool({
  name: "change_pytorch_target_column",
  description: "Change the PyTorch target column.",
  parameters: z.object({
    target_column: z.string().optional(),
    mode: z.enum(["different", "random", "next"]).optional(),
  }),
  handler: async (args) => {
    await ensureTab("pytorch");
    const nextTarget = resolveTargetColumnChangeFromOptions(getSelectOptions(PYTORCH_FIELD_SELECTORS.target_column), args);
    if (!nextTarget) return formatChangeTargetColumnToolResult("PyTorch", undefined, { status: "error", code: "PYTORCH_TARGET_COLUMN_UNAVAILABLE" });
    const bridge = getPytorchBridge();
    if (!bridge) return "Unable to change PyTorch target column: PYTORCH_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch({ target_column: nextTarget });
    return formatChangeTargetColumnToolResult("PyTorch", nextTarget, result);
  },
});

useFrontendTool({
  name: "randomize_pytorch_form_fields",
  description: "Randomize PyTorch form fields with validator-safe values.",
  parameters: z.object({
    confirm_randomize: z.boolean(),
    value_count: z.number().optional(),
    style: z.enum(["safe", "balanced", "aggressive"]).optional(),
    set_sweep_values: z.boolean().optional(),
    run_sweep: z.boolean().optional(),
    auto_distill: z.boolean().optional(),
    randomize_model_fields: z.boolean().optional(),
  }),
  handler: async (args) => {
    if (!args.confirm_randomize) return JSON.stringify({ status: "error", code: "RANDOMIZE_CONFIRMATION_REQUIRED" });
    await ensureTab("pytorch");
    const patch = buildRandomPytorchFormPatch(args, {
      getSelectValue: (f) => f === "pytorch_training_mode" ? getSelectValue(PYTORCH_FIELD_SELECTORS.training_mode) : f === "pytorch_task" ? getSelectValue(PYTORCH_FIELD_SELECTORS.task) : null,
    });
    const bridge = getPytorchBridge();
    if (!bridge) return "Unable to randomize PyTorch form: PYTORCH_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch(patch);
    return formatRandomizeFormFieldsToolResult("PyTorch", { status: "ok", randomized: true, patch, ...result });
  },
});

useFrontendTool({
  name: "start_pytorch_training_runs",
  description: "Start PyTorch training runs using the currently configured form values.",
  parameters: z.object({}),
  handler: async () => {
    await ensureTab("pytorch");
    const bridge = getPytorchBridge();
    if (!bridge?.startTrainingRuns) return "Unable to start PyTorch training: PYTORCH_FORM_UNAVAILABLE.";
    const result = await bridge.startTrainingRuns();
    return formatStartTrainingRunsToolResult("PyTorch", result);
  },
});

useFrontendTool({
  name: "train_pytorch_model",
  description: "Execute a direct backend PyTorch training request with explicit parameters.",
  parameters: z.object({
    dataset_id: z.string(),
    target_column: z.string(),
    task: z.string().optional(),
    epochs: z.number().optional(),
    batch_size: z.number().optional(),
    learning_rate: z.number().optional(),
  }),
  handler: async (args) => {
    const result = await mlApi.trainPytorch({
      dataset_id: args.dataset_id,
      target_column: args.target_column,
      task: args.task ?? "auto",
      epochs: args.epochs ?? 60,
      batch_size: args.batch_size ?? 64,
      learning_rate: args.learning_rate ?? 0.001,
      training_mode: "mlp_dense",
      exclude_columns: [],
      date_columns: [],
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
    });
    return formatTrainModelToolResult("PyTorch", result as Record<string, unknown>);
  },
});

// --- TensorFlow-specific Tools ---

useFrontendTool({
  name: "set_tensorflow_form_fields",
  description: "Set/patch TensorFlow training form fields on the /ag-ui TensorFlow tab.",
  parameters: z.object({ fields: z.record(z.any()) }),
  handler: async (args) => {
    const patch = resolveMlFormPatchFromToolArgs(args);
    await ensureTab("tensorflow");
    const bridge = getTensorflowBridge();
    if (!bridge) return "Unable to update TensorFlow form fields: TENSORFLOW_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch(patch);
    return formatSetFormFieldsToolResult("TensorFlow", result);
  },
});

useFrontendTool({
  name: "change_tensorflow_target_column",
  description: "Change the TensorFlow target column.",
  parameters: z.object({
    target_column: z.string().optional(),
    mode: z.enum(["different", "random", "next"]).optional(),
  }),
  handler: async (args) => {
    await ensureTab("tensorflow");
    const nextTarget = resolveTargetColumnChangeFromOptions(getSelectOptions(TENSORFLOW_FIELD_SELECTORS.target_column), args);
    if (!nextTarget) return formatChangeTargetColumnToolResult("TensorFlow", undefined, { status: "error", code: "TENSORFLOW_TARGET_COLUMN_UNAVAILABLE" });
    const bridge = getTensorflowBridge();
    if (!bridge) return "Unable to change TensorFlow target column: TENSORFLOW_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch({ target_column: nextTarget });
    return formatChangeTargetColumnToolResult("TensorFlow", nextTarget, result);
  },
});

useFrontendTool({
  name: "randomize_tensorflow_form_fields",
  description: "Randomize TensorFlow form fields with validator-safe values.",
  parameters: z.object({
    confirm_randomize: z.boolean(),
    value_count: z.number().optional(),
    style: z.enum(["safe", "balanced", "aggressive"]).optional(),
    set_sweep_values: z.boolean().optional(),
    run_sweep: z.boolean().optional(),
    auto_distill: z.boolean().optional(),
    randomize_model_fields: z.boolean().optional(),
  }),
  handler: async (args) => {
    if (!args.confirm_randomize) return JSON.stringify({ status: "error", code: "RANDOMIZE_CONFIRMATION_REQUIRED" });
    await ensureTab("tensorflow");
    const patch = buildRandomTensorflowFormPatch(args, {
      getSelectValue: (f) => f === "tensorflow_task" ? getSelectValue(TENSORFLOW_FIELD_SELECTORS.task) : null,
    });
    const bridge = getTensorflowBridge();
    if (!bridge) return "Unable to randomize TensorFlow form: TENSORFLOW_FORM_UNAVAILABLE.";
    const result = bridge.applyPatch(patch);
    return formatRandomizeFormFieldsToolResult("TensorFlow", { status: "ok", randomized: true, patch, ...result });
  },
});

useFrontendTool({
  name: "start_tensorflow_training_runs",
  description: "Start TensorFlow training runs using the currently configured form values.",
  parameters: z.object({}),
  handler: async () => {
    await ensureTab("tensorflow");
    const bridge = getTensorflowBridge();
    if (!bridge?.startTrainingRuns) return "Unable to start TensorFlow training: TENSORFLOW_FORM_UNAVAILABLE.";
    const result = await bridge.startTrainingRuns();
    return formatStartTrainingRunsToolResult("TensorFlow", result);
  },
});

useFrontendTool({
  name: "train_tensorflow_model",
  description: "Execute a direct backend TensorFlow training request with explicit parameters.",
  parameters: z.object({
    dataset_id: z.string(),
    target_column: z.string(),
    task: z.string().optional(),
    epochs: z.number().optional(),
    batch_size: z.number().optional(),
    learning_rate: z.number().optional(),
  }),
  handler: async (args) => {
    const result = await mlApi.trainTensorflow({
      dataset_id: args.dataset_id,
      target_column: args.target_column,
      task: args.task ?? "auto",
      epochs: args.epochs ?? 60,
      batch_size: args.batch_size ?? 64,
      learning_rate: args.learning_rate ?? 0.001,
      training_mode: "wide_and_deep",
      exclude_columns: [],
      date_columns: [],
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
    });
    return formatTrainModelToolResult("TensorFlow", result as Record<string, unknown>);
  },
});

// --- Readable Context ---

useAgentContext({
  description: "ag_ui_active_tab",
  value: toRef(() => props.activeTab),
});

useAgentContext({
  description: "agentic_research_selected_dataset_id",
  value: toRef(() => props.activeDatasetId ?? ""),
});

useAgentContext({
  description: "ag_ui_selected_model_id",
  value: toRef(() => props.selectedModelId || ""),
});
</script>
