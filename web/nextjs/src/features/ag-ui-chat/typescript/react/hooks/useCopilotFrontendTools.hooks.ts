"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useRouter } from "next/navigation";
import type {
  NavigateToPageArgs,
  RandomizePytorchFormFieldsArgs,
  RandomizeTensorflowFormFieldsArgs,
  TrainTensorflowModelArgs,
  TrainPytorchModelArgs,
} from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotTools.types";
import type {
  CopilotFrontendToolsRuntime,
} from "@/features/ag-ui-chat/__types__/typescript/logic/copilotFrontendToolsFlow.types";
import { useCopilotChartActionsAdapter } from "@/features/recharts/typescript/react/ai/state/adapters/chartActions.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import {
  handleAddChartSpec,
  handleNavigateToPage,
  resolvePytorchFormPatchFromToolArgs,
  resolveTensorflowFormPatchFromToolArgs,
} from "@/features/ag-ui-chat/typescript/logic/frontendTools.logic";
import {
  handleRandomizePytorchFormFields,
  handleRandomizeTensorflowFormFields,
  handleSetPytorchFormFields,
  handleSetTensorflowFormFields,
  handleStartTensorflowTrainingRuns,
  handleStartPytorchTrainingRuns,
  handleTrainTensorflowModel,
  handleTrainPytorchModel,
} from "@/features/ag-ui-chat/typescript/logic/frontendTools.adapter";
import {
  ensurePytorchTab,
  ensureTensorflowTab,
  waitForPytorchForm,
  waitForTensorflowForm,
} from "@/features/ag-ui-chat/typescript/logic/copilotFrontendToolsFlow.logic";
import { createCopilotFrontendToolActions } from "@/features/ag-ui-chat/typescript/logic/copilotFrontendToolActions.logic";
import { ML_TRAINING_FRAMEWORKS } from "@/features/ml/typescript/ai/agUi/mlTrainingFrameworkMetadata.logic";
import type {
  MlFormPatch,
  MlFormRandomizeArgs,
  MlFrameworkTab,
} from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";

function defaultNextFrame(): Promise<void> {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

type FrameworkActionBundle<TTrainArgs, TRandomizeArgs> = {
  startTrainingRuns: () => Promise<unknown>;
  trainModel: (args: TTrainArgs) => Promise<unknown>;
  setFormFields: (args: Record<string, unknown>) => Promise<unknown>;
  randomizeFormFields: (args: TRandomizeArgs) => Promise<unknown>;
};

function useRegisterFrameworkCopilotActions(
  actions: {
    startTrainingRuns: unknown;
    trainModel: unknown;
    setFormFields: unknown;
    randomizeFormFields: unknown;
  },
  deps: readonly unknown[]
): void {
  useCopilotAction(
    actions.startTrainingRuns as Parameters<typeof useCopilotAction>[0],
    [...deps]
  );
  useCopilotAction(actions.trainModel as Parameters<typeof useCopilotAction>[0], []);
  useCopilotAction(
    actions.setFormFields as Parameters<typeof useCopilotAction>[0],
    [...deps]
  );
  useCopilotAction(
    actions.randomizeFormFields as Parameters<typeof useCopilotAction>[0],
    [...deps]
  );
}

/**
 * Registers AG-UI frontend tool calls in a hook (view stays thin).
 */
export function useCopilotFrontendTools(runtime: CopilotFrontendToolsRuntime = {}): void {
  const { addChartSpec: addGlobalChartSpec, clearChartSpecs: clearGlobalCharts } =
    useCopilotChartActionsAdapter();
  const { addChartSpec: addAgenticChartSpec, clearChartSpecs: clearAgenticCharts } =
    useAgenticResearchChartActionsAdapter();
  const { activeTab, setActiveTab } = useAgUiWorkspaceStateAdapter();
  const router = useRouter();
  const nextFrame = runtime.nextFrame ?? defaultNextFrame;

  const ensureFrameworkTab = async (framework: MlFrameworkTab): Promise<void> => {
    if (framework === ML_TRAINING_FRAMEWORKS.pytorch.tab) {
      await ensurePytorchTab({
        activeTab,
        setActiveTab,
        pushRoute: router.push,
        waitForPytorchForm: () => waitForPytorchForm(1800, runtime),
      });
      return;
    }

    await ensureTensorflowTab({
      activeTab,
      setActiveTab,
      pushRoute: router.push,
      waitForTensorflowForm: () => waitForTensorflowForm(1800, runtime),
    });
  };

  function createFrameworkHandlers<TTrainArgs, TRandomizeArgs extends MlFormRandomizeArgs, TPatch extends MlFormPatch>(
    framework: MlFrameworkTab,
    resolvePatch: (args: Record<string, unknown>) => TPatch,
    startTrainingRuns: () => Promise<unknown>,
    trainModel: (args: TTrainArgs) => Promise<unknown>,
    setFormFields: (patch: TPatch) => { status: "ok" | "error"; [key: string]: unknown },
    randomizeFormFields: (args: TRandomizeArgs) => unknown
  ): FrameworkActionBundle<TTrainArgs, TRandomizeArgs> {
    return {
      startTrainingRuns: async () => {
        await ensureFrameworkTab(framework);
        return startTrainingRuns();
      },
      trainModel: async (args: TTrainArgs) => trainModel(args),
      setFormFields: async (args: Record<string, unknown>) => {
        const patch = resolvePatch(args);
        await ensureFrameworkTab(framework);
        const result = setFormFields(patch);
        if (result.status === "ok") {
          await (runtime.delay ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))))(0);
          await nextFrame();
        }
        return result;
      },
      randomizeFormFields: async (args: TRandomizeArgs) => {
        await ensureFrameworkTab(framework);
        return randomizeFormFields(args);
      },
    };
  }

  const pytorchHandlers = createFrameworkHandlers<
    TrainPytorchModelArgs,
    RandomizePytorchFormFieldsArgs,
    ReturnType<typeof resolvePytorchFormPatchFromToolArgs>
  >(
    "pytorch",
    resolvePytorchFormPatchFromToolArgs,
    handleStartPytorchTrainingRuns,
    handleTrainPytorchModel,
    handleSetPytorchFormFields,
    handleRandomizePytorchFormFields
  );

  const tensorflowHandlers = createFrameworkHandlers<
    TrainTensorflowModelArgs,
    RandomizeTensorflowFormFieldsArgs,
    ReturnType<typeof resolveTensorflowFormPatchFromToolArgs>
  >(
    "tensorflow",
    resolveTensorflowFormPatchFromToolArgs,
    handleStartTensorflowTrainingRuns,
    handleTrainTensorflowModel,
    handleSetTensorflowFormFields,
    handleRandomizeTensorflowFormFields
  );

  const actions = createCopilotFrontendToolActions({
    handleAddChartSpec: ({ chartSpec, chartSpecs }) => {
      const targetAdd = activeTab === "agentic-research" ? addAgenticChartSpec : addGlobalChartSpec;
      return handleAddChartSpec({ chartSpec, chartSpecs }, targetAdd);
    },
    handleClearCharts: () => {
      if (activeTab === "agentic-research") {
        clearAgenticCharts();
      } else {
        clearGlobalCharts();
      }
      return { status: "ok" as const, cleared: true };
    },
    handleNavigateToPage: ({ route }: NavigateToPageArgs) => {
      const result = handleNavigateToPage(route);
      if (result.status === "ok") router.push(result.resolvedRoute);
      return result;
    },
    handleStartPytorchTrainingRuns: pytorchHandlers.startTrainingRuns,
    handleTrainPytorchModel: pytorchHandlers.trainModel,
    handleSetPytorchFormFields: pytorchHandlers.setFormFields,
    handleRandomizePytorchFormFields: pytorchHandlers.randomizeFormFields,
    handleStartTensorflowTrainingRuns: tensorflowHandlers.startTrainingRuns,
    handleTrainTensorflowModel: tensorflowHandlers.trainModel,
    handleSetTensorflowFormFields: tensorflowHandlers.setFormFields,
    handleRandomizeTensorflowFormFields: tensorflowHandlers.randomizeFormFields,
  });

  useCopilotAction(
    actions.addChartSpec as Parameters<typeof useCopilotAction>[0],
    [activeTab, addAgenticChartSpec, addGlobalChartSpec]
  );

  useCopilotAction(
    actions.clearCharts as Parameters<typeof useCopilotAction>[0],
    [activeTab, clearAgenticCharts, clearGlobalCharts]
  );

  useCopilotAction(
    actions.navigateToPage as Parameters<typeof useCopilotAction>[0],
    [router]
  );

  const frameworkActionDeps = [activeTab, router, setActiveTab] as const;
  useRegisterFrameworkCopilotActions(
    {
      startTrainingRuns: actions.startPytorchTrainingRuns,
      trainModel: actions.trainPytorchModel,
      setFormFields: actions.setPytorchFormFields,
      randomizeFormFields: actions.randomizePytorchFormFields,
    },
    frameworkActionDeps
  );
  useRegisterFrameworkCopilotActions(
    {
      startTrainingRuns: actions.startTensorflowTrainingRuns,
      trainModel: actions.trainTensorflowModel,
      setFormFields: actions.setTensorflowFormFields,
      randomizeFormFields: actions.randomizeTensorflowFormFields,
    },
    frameworkActionDeps
  );
}
