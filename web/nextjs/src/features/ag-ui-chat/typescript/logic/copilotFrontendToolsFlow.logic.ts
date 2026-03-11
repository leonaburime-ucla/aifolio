import type {
  CopilotFrontendToolsRuntime,
  EnsurePytorchTabArgs,
  EnsureTensorflowTabArgs,
} from "@/features/ag-ui-chat/__types__/typescript/logic/copilotFrontendToolsFlow.types";
import {
  ensureFrameworkTab,
  waitForFrameworkFormField,
} from "@/features/ml/typescript/ai/agUi/mlTrainingToolsFlow.logic";
import { ML_TRAINING_FRAMEWORKS } from "@/features/ml/typescript/ai/agUi/mlTrainingFrameworkMetadata.logic";
import type {
  PytorchFormBridge,
  TensorflowFormBridge,
} from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";

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

export async function waitForPytorchForm(
  timeoutMs = 1800,
  runtime: CopilotFrontendToolsRuntime = {}
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

export async function waitForTensorflowForm(
  timeoutMs = 1800,
  runtime: CopilotFrontendToolsRuntime = {}
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
