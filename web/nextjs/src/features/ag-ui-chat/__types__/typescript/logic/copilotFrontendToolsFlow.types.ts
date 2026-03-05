import type { EnsureFrameworkTabArgs } from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";

export type CopilotFrontendToolsRuntime = {
  querySelector?: (selector: string) => Element | null;
  delay?: (ms: number) => Promise<void>;
  nextFrame?: () => Promise<void>;
};

export type EnsurePytorchTabArgs = Omit<EnsureFrameworkTabArgs, "frameworkTab" | "waitForFrameworkForm"> & {
  waitForPytorchForm: () => Promise<boolean>;
};

export type EnsureTensorflowTabArgs = Omit<EnsureFrameworkTabArgs, "frameworkTab" | "waitForFrameworkForm"> & {
  waitForTensorflowForm: () => Promise<boolean>;
};
