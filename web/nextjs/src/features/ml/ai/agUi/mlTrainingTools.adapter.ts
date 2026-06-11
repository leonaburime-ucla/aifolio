import type {
  ChangePytorchTargetColumnArgs,
  ChangeTensorflowTargetColumnArgs,
  PytorchFormBridge,
  PytorchFormPatch,
  PytorchRandomizeArgs,
  TensorflowFormBridge,
  TensorflowFormPatch,
  TensorflowRandomizeArgs,
} from "@aifolio/contracts/entities/ml-training";
import {
  buildRandomPytorchFormPatch as buildRandomPytorchFormPatchCore,
  buildRandomTensorflowFormPatch as buildRandomTensorflowFormPatchCore,
  PYTORCH_FIELD_SELECTORS,
  resolveTargetColumnChangeFromOptions,
  TENSORFLOW_FIELD_SELECTORS,
} from "@aifolio/frontend-core/ag-ui";
import { toBridgeCsv } from "@aifolio/frontend-core/ml-training";
import { trainPytorchModel, type PytorchTrainRequest } from "@/features/ml/api/pytorchApi";
import { trainTensorflowModel, type TensorflowTrainRequest } from "@/features/ml/api/tensorflowApi";

/**
 * ML AG-UI training tool adapter layer.
 *
 * Purpose:
 * - Encapsulate browser/runtime side effects for framework training tool calls.
 * - Bridge pure training logic with form state and backend APIs.
 *
 * Layering:
 * - This file is intentionally adapter-bound (window/document/network).
 * - Pure transformations/validation stay in `mlTrainingTools.logic.ts`.
 */

/**
 * Resolves the optional global PyTorch form bridge from `window`.
 *
 * @returns Bridge when available in browser runtime, otherwise `null`.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
function getPytorchBridge(): PytorchFormBridge | null {
  if (typeof window === "undefined") return null;
  const bridge = (window as Window & { __AIFOLIO_PYTORCH_FORM_BRIDGE__?: PytorchFormBridge })
    .__AIFOLIO_PYTORCH_FORM_BRIDGE__;
  return bridge ?? null;
}

/**
 * Resolves the optional global TensorFlow form bridge from `window`.
 *
 * @returns Bridge when available in browser runtime, otherwise `null`.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
function getTensorflowBridge(): TensorflowFormBridge | null {
  if (typeof window === "undefined") return null;
  const bridge = (window as Window & { __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: TensorflowFormBridge })
    .__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  return bridge ?? null;
}

/**
 * Finds a form control by selector in browser context.
 *
 * @param selector CSS selector for the target control.
 * @returns Matching DOM element or `null`.
 * @complexity O(d) DOM selector lookup cost controlled by the browser engine, O(1) space.
 * @overallScore 100
 */
function findControl(selector: string): Element | null {
  if (typeof document === "undefined") return null;
  return document.querySelector(selector);
}

/**
 * Applies a value to a control and dispatches synthetic events so React state
 * observes the mutation.
 *
 * @param selector Control selector.
 * @param value Value to apply.
 * @returns `true` when the value was applied; otherwise `false`.
 * @complexity O(o) for select option validation, otherwise O(1); O(o) transient space for select options.
 * @overallScore 100
 */
function setControlValue(selector: string, value: unknown): boolean {
  if (typeof document === "undefined") return false;
  const element = findControl(selector);
  if (!element) return false;

  if (element instanceof HTMLInputElement) {
    if (element.type === "checkbox") {
      element.checked = Boolean(value);
      element.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    }
    element.value = toBridgeCsv(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  if (element instanceof HTMLSelectElement) {
    const nextValue = toBridgeCsv(value);
    const hasOption = Array.from(element.options).some((option) => option.value === nextValue);
    if (!hasOption) return false;
    element.value = nextValue;
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  return false;
}

/**
 * Returns ordered non-empty select values from a form control.
 *
 * @param selector Select control selector.
 * @returns Available values in DOM order.
 * @complexity O(o) time and space for o select options.
 * @overallScore 100
 */
function getSelectOptions(selector: string) {
  const select = findControl(selector);
  if (!(select instanceof HTMLSelectElement)) return [];
  const currentValue = select.value?.trim();
  return Array.from(select.options)
    .map((option) => option.value)
    .filter((value) => value && value.trim().length > 0)
    .map((value) => ({ value, isCurrent: value === currentValue }));
}

/**
 * Resolves a target-column change request into a concrete target value.
 *
 * Explicit `target_column` wins. Otherwise the tool defaults to a different
 * target so "change the target column" remains deterministic.
 *
 * @param selector Target-column select selector.
 * @param args Requested target-column change args.
 * @returns Concrete target column or `null` when no valid change is possible.
 * @complexity O(o) time and space for o select options.
 * @overallScore 100
 */
function resolveTargetColumnChange(
  selector: string,
  args: { target_column?: string; mode?: "different" | "random" | "next" }
): string | null {
  return resolveTargetColumnChangeFromOptions(getSelectOptions(selector), args);
}

/**
 * Reads a trimmed selected value from a `<select>` control.
 *
 * @param selector Control selector.
 * @returns Selected value or `null` when unavailable/empty.
 * @complexity O(1) time and space after selector lookup.
 * @overallScore 100
 */
function getSelectValue(selector: string): string | null {
  const element = findControl(selector);
  if (!(element instanceof HTMLSelectElement)) return null;
  const value = element.value?.trim();
  return value ? value : null;
}

/**
 * Executes a direct backend PyTorch training request.
 *
 * @param payload Backend request body for a single PyTorch training run.
 * @returns API result from `trainPytorchModel`.
 * @complexity O(1) local work, excluding network latency and response size.
 * @overallScore 100
 */
export async function handleTrainPytorchModel(payload: PytorchTrainRequest) {
  return trainPytorchModel(payload);
}

/**
 * Executes a direct backend TensorFlow training request.
 *
 * @param payload Backend request body for a single TensorFlow training run.
 * @returns API result from `trainTensorflowModel`.
 * @complexity O(1) local work, excluding network latency and response size.
 * @overallScore 100
 */
export async function handleTrainTensorflowModel(payload: TensorflowTrainRequest) {
  return trainTensorflowModel(payload);
}

/**
 * Applies a partial PyTorch form patch through bridge first, then DOM fallback.
 *
 * @param patch Partial field patch for PyTorch form controls.
 * @returns Applied/skipped field report or an availability/validation error.
 * @complexity O(k * o) worst-case DOM fallback over k patch keys and o select options; O(k) space.
 * @overallScore 100
 */
export function handleSetPytorchFormFields(patch: PytorchFormPatch) {
  if (typeof document === "undefined") {
    return { status: "error" as const, code: "PYTORCH_FORM_UNAVAILABLE" as const };
  }

  const bridge = getPytorchBridge();
  if (bridge) {
    const result = bridge.applyPatch(patch);
    if (!result.applied.length) {
      return {
        status: "error" as const,
        code: "NO_FIELDS_APPLIED" as const,
        skipped: result.skipped,
      };
    }
    return {
      status: "ok" as const,
      applied: result.applied,
      skipped: result.skipped,
      via: "bridge" as const,
    };
  }

  const applied: string[] = [];
  const skipped: string[] = [];
  const orderedKeys: Array<keyof typeof PYTORCH_FIELD_SELECTORS> = [
    "run_sweep",
    "training_mode",
    "target_column",
    "task",
    "epoch_values",
    "batch_sizes",
    "learning_rates",
    "test_sizes",
    "hidden_dims",
    "num_hidden_layers",
    "dropouts",
    "exclude_columns",
    "date_columns",
    "auto_distill",
  ];

  orderedKeys.forEach((key) => {
    const value = patch[key];
    if (value === undefined) return;
    const selector = PYTORCH_FIELD_SELECTORS[key];
    if (!selector) return;
    const ok = setControlValue(selector, value);
    if (ok) applied.push(String(key));
    else skipped.push(String(key));
  });

  if (!applied.length) {
    return {
      status: "error" as const,
      code: "NO_FIELDS_APPLIED" as const,
      skipped,
    };
  }

  return {
    status: "ok" as const,
    applied,
    skipped,
  };
}

/**
 * Applies a partial TensorFlow form patch through bridge.
 *
 * @param patch Partial field patch for TensorFlow form controls.
 * @returns Applied/skipped field report or an availability/validation error.
 * @complexity O(k) time and space for bridge-applied patch keys.
 * @overallScore 100
 */
export function handleSetTensorflowFormFields(patch: TensorflowFormPatch) {
  if (typeof document === "undefined") {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }

  const bridge = getTensorflowBridge();
  if (!bridge) {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }

  const result = bridge.applyPatch(patch);
  if (!result.applied.length) {
    return {
      status: "error" as const,
      code: "NO_FIELDS_APPLIED" as const,
      skipped: result.skipped,
    };
  }
  return {
    status: "ok" as const,
    applied: result.applied,
    skipped: result.skipped,
    via: "bridge" as const,
  };
}

/**
 * Builds a validator-safe randomized PyTorch form patch from current DOM state.
 *
 * @param args Optional randomization options controlling aggressiveness and field locking.
 * @returns Patch suitable for `handleSetPytorchFormFields`.
 * @complexity O(v) time and space for selected sweep values plus constant DOM reads.
 * @overallScore 100
 */
export function buildRandomPytorchFormPatch(args: PytorchRandomizeArgs = {}): PytorchFormPatch {
  return buildRandomPytorchFormPatchCore(args, {
    getSelectValue: (field) => {
      if (field === "pytorch_training_mode") {
        return getSelectValue(PYTORCH_FIELD_SELECTORS.training_mode);
      }
      if (field === "pytorch_task") {
        return getSelectValue(PYTORCH_FIELD_SELECTORS.task);
      }
      return null;
    },
  });
}

/**
 * Randomizes PyTorch form fields and immediately applies the patch.
 *
 * @param args Optional randomization behavior controls.
 * @returns Randomization/apply status plus applied patch details.
 * @complexity O(k * o) worst-case DOM fallback over k patch keys and o select options; O(k) space.
 * @overallScore 100
 */
export function handleRandomizePytorchFormFields(args: PytorchRandomizeArgs = {}) {
  if (args.confirm_randomize !== true) {
    return {
      status: "error" as const,
      code: "RANDOMIZE_CONFIRMATION_REQUIRED" as const,
      message: "Randomization blocked. Pass confirm_randomize=true to randomize form fields.",
    };
  }
  const patch = buildRandomPytorchFormPatch(args);
  const applied = handleSetPytorchFormFields(patch);
  if (applied.status !== "ok") {
    return applied;
  }
  return {
    status: "ok" as const,
    randomized: true,
    patch,
    applied: applied.applied,
    skipped: applied.skipped,
  };
}

/**
 * Builds a validator-safe randomized TensorFlow form patch from current DOM state.
 *
 * @param args Optional randomization options controlling aggressiveness and field locking.
 * @returns Patch suitable for `handleSetTensorflowFormFields`.
 * @complexity O(v) time and space for selected sweep values plus constant DOM reads.
 * @overallScore 100
 */
export function buildRandomTensorflowFormPatch(args: TensorflowRandomizeArgs = {}): TensorflowFormPatch {
  return buildRandomTensorflowFormPatchCore(args, {
    getSelectValue: (field) => {
      if (field === "tensorflow_task") {
        return getSelectValue(TENSORFLOW_FIELD_SELECTORS.task);
      }
      return null;
    },
  });
}

/**
 * Randomizes TensorFlow form fields and immediately applies the patch.
 *
 * @param args Optional randomization behavior controls.
 * @returns Randomization/apply status plus applied patch details.
 * @complexity O(k) time and space for bridge-applied patch keys.
 * @overallScore 100
 */
export function handleRandomizeTensorflowFormFields(args: TensorflowRandomizeArgs = {}) {
  if (args.confirm_randomize !== true) {
    return {
      status: "error" as const,
      code: "RANDOMIZE_CONFIRMATION_REQUIRED" as const,
      message: "Randomization blocked. Pass confirm_randomize=true to randomize form fields.",
    };
  }
  const patch = buildRandomTensorflowFormPatch(args);
  const applied = handleSetTensorflowFormFields(patch);
  if (applied.status !== "ok") {
    return applied;
  }
  return {
    status: "ok" as const,
    randomized: true,
    patch,
    applied: applied.applied,
    skipped: applied.skipped,
  };
}

/**
 * Changes the PyTorch target column through the existing form bridge.
 *
 * @param args Optional explicit target or selection mode.
 * @returns Applied target-column change or an availability/validation error.
 * @complexity O(o + k) time and O(o + k) space for select options and applied keys.
 * @overallScore 100
 */
export function handleChangePytorchTargetColumn(args: ChangePytorchTargetColumnArgs = {}) {
  const nextTarget = resolveTargetColumnChange(PYTORCH_FIELD_SELECTORS.target_column, args);
  if (!nextTarget) {
    return { status: "error" as const, code: "PYTORCH_TARGET_COLUMN_UNAVAILABLE" as const };
  }
  return handleSetPytorchFormFields({ target_column: nextTarget });
}

/**
 * Changes the TensorFlow target column through the existing form bridge.
 *
 * @param args Optional explicit target or selection mode.
 * @returns Applied target-column change or an availability/validation error.
 * @complexity O(o + k) time and O(o + k) space for select options and applied keys.
 * @overallScore 100
 */
export function handleChangeTensorflowTargetColumn(args: ChangeTensorflowTargetColumnArgs = {}) {
  const nextTarget = resolveTargetColumnChange(TENSORFLOW_FIELD_SELECTORS.target_column, args);
  if (!nextTarget) {
    return { status: "error" as const, code: "TENSORFLOW_TARGET_COLUMN_UNAVAILABLE" as const };
  }
  return handleSetTensorflowFormFields({ target_column: nextTarget });
}

/**
 * Starts PyTorch training runs via the injected window bridge on the PyTorch page.
 *
 * @returns Success when training was started; otherwise a typed error code.
 * @complexity O(1) local work, excluding injected bridge runtime.
 * @overallScore 100
 */
export async function handleStartPytorchTrainingRuns() {
  const bridge = getPytorchBridge();
  if (!bridge?.startTrainingRuns) {
    return { status: "error" as const, code: "PYTORCH_FORM_UNAVAILABLE" as const };
  }
  const result = await bridge.startTrainingRuns();
  if (result.status === "ok") {
    return { status: "ok" as const, started: true };
  }
  return { status: "error" as const, code: result.reason };
}

/**
 * Starts TensorFlow training runs via the injected window bridge on the TensorFlow page.
 *
 * @returns Success when training was started; otherwise a typed error code.
 * @complexity O(1) local work, excluding injected bridge runtime.
 * @overallScore 100
 */
export async function handleStartTensorflowTrainingRuns() {
  const bridge = getTensorflowBridge();
  if (!bridge?.startTrainingRuns) {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }
  const result = await bridge.startTrainingRuns();
  if (result.status === "ok") {
    return { status: "ok" as const, started: true };
  }
  return { status: "error" as const, code: result.reason };
}
