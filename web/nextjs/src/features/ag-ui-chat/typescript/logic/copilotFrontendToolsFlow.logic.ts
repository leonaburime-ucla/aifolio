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

/**
 * AG-UI wrappers over ML-owned framework flow primitives.
 */

export async function waitForPytorchForm(
  timeoutMs = 1800,
  runtime: CopilotFrontendToolsRuntime = {}
): Promise<boolean> {
  return waitForFrameworkFormField(ML_TRAINING_FRAMEWORKS.pytorch.targetSelector, timeoutMs, runtime);
}

export async function waitForTensorflowForm(
  timeoutMs = 1800,
  runtime: CopilotFrontendToolsRuntime = {}
): Promise<boolean> {
  return waitForFrameworkFormField(ML_TRAINING_FRAMEWORKS.tensorflow.targetSelector, timeoutMs, runtime);
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
