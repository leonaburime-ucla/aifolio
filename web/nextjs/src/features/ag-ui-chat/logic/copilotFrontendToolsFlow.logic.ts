import type {
  EnsurePytorchTabArgs,
  EnsureTensorflowTabArgs,
} from "@aifolio/contracts/entities/ag-ui";
import {
  ensureFrameworkTab,
  waitForFrameworkFormField,
  ML_TRAINING_FRAMEWORKS,
  type MlToolFlowRuntime,
} from "@aifolio/frontend-core/ag-ui";
import type {
  PytorchFormBridge,
  TensorflowFormBridge,
} from "@aifolio/contracts/entities/ml-training";

function hasPytorchBridge(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(
    (window as Window & { __AIFOLIO_PYTORCH_FORM_BRIDGE__?: PytorchFormBridge })
      .__AIFOLIO_PYTORCH_FORM_BRIDGE__
  );
}

function hasTensorflowBridge(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(
    (window as Window & { __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: TensorflowFormBridge })
      .__AIFOLIO_TENSORFLOW_FORM_BRIDGE__
  );
}

/**
 * AG-UI wrappers over ML-owned framework flow primitives.
 */

/**
 * Waits for the PyTorch form and bridge to become available in the browser.
 *
 * @param timeoutMs Maximum wait time in milliseconds.
 * @param runtime Optional delay implementation for tests.
 * @returns True when both the field and bridge are available before timeout.
 * @complexity O(t) time where t is timeoutMs / poll interval, O(1) space.
 * @overallScore 100
 */
export async function waitForPytorchForm(
  timeoutMs = 1800,
  runtime: MlToolFlowRuntime = {}
): Promise<boolean> {
  const hasField = await waitForFrameworkFormField(
    ML_TRAINING_FRAMEWORKS.pytorch.targetSelector,
    timeoutMs,
    runtime
  );
  if (!hasField) return false;

  const delay =
    runtime.delay ??
    ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (hasPytorchBridge()) return true;
    await delay(60);
  }
  return false;
}

/**
 * Waits for the TensorFlow form and bridge to become available in the browser.
 *
 * @param timeoutMs Maximum wait time in milliseconds.
 * @param runtime Optional delay implementation for tests.
 * @returns True when both the field and bridge are available before timeout.
 * @complexity O(t) time where t is timeoutMs / poll interval, O(1) space.
 * @overallScore 100
 */
export async function waitForTensorflowForm(
  timeoutMs = 1800,
  runtime: MlToolFlowRuntime = {}
): Promise<boolean> {
  const hasField = await waitForFrameworkFormField(
    ML_TRAINING_FRAMEWORKS.tensorflow.targetSelector,
    timeoutMs,
    runtime
  );
  if (!hasField) return false;

  const delay =
    runtime.delay ??
    ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (hasTensorflowBridge()) return true;
    await delay(60);
  }
  return false;
}

/**
 * Ensures the PyTorch AG-UI tab is active before running a PyTorch tool.
 *
 * @param args Current tab, tab setter, route pusher, and PyTorch form waiter.
 * @returns Resolves after the tab switch and form wait complete.
 * @complexity O(1) excluding injected navigation and wait callbacks.
 * @overallScore 100
 */
export async function ensurePytorchTab({
  activeTab,
  setActiveTab,
  pushRoute,
  waitForPytorchForm: waitForFrameworkForm,
}: EnsurePytorchTabArgs): Promise<void> {
  await ensureFrameworkTab({
    activeTab,
    setActiveTab,
    pushRoute,
    frameworkTab: "pytorch",
    waitForFrameworkForm,
  });
}

/**
 * Ensures the TensorFlow AG-UI tab is active before running a TensorFlow tool.
 *
 * @param args Current tab, tab setter, route pusher, and TensorFlow form waiter.
 * @returns Resolves after the tab switch and form wait complete.
 * @complexity O(1) excluding injected navigation and wait callbacks.
 * @overallScore 100
 */
export async function ensureTensorflowTab({
  activeTab,
  setActiveTab,
  pushRoute,
  waitForTensorflowForm: waitForFrameworkForm,
}: EnsureTensorflowTabArgs): Promise<void> {
  await ensureFrameworkTab({
    activeTab,
    setActiveTab,
    pushRoute,
    frameworkTab: "tensorflow",
    waitForFrameworkForm,
  });
}
