"use client";

import { useMemo } from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import { useRouter } from "next/navigation";
import type {
  NavigateToPageArgs,
} from "@aifolio/contracts/entities/ag-ui";
import type {
  PytorchRandomizeArgs as RandomizePytorchFormFieldsArgs,
  TensorflowRandomizeArgs as RandomizeTensorflowFormFieldsArgs,
  TrainPytorchModelArgs,
  TrainTensorflowModelArgs,
} from "@aifolio/contracts/entities/ml-training";
import { useCopilotChartActionsAdapter } from "@/features/recharts/react/ai/state/adapters/chartActions.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/react/state/adapters/chartActions.adapter";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/react/state/adapters/agUiWorkspaceState.adapter";
import { useAgUiWorkspaceStore } from "@/features/ag-ui-chat/react/state/zustand/agUiWorkspaceStore";
import {
  handleAddChartSpec,
  handleNavigateToPage,
  resolveMlFormPatchFromToolArgs,
  formatAddChartSpecToolResult,
  formatChangeTargetColumnToolResult,
  formatClearChartsToolResult,
  formatNavigateToPageToolResult,
  formatRandomizeFormFieldsToolResult,
  formatSetFormFieldsToolResult,
  formatStartTrainingRunsToolResult,
  formatTrainModelToolResult,
  ML_TRAINING_FRAMEWORKS,
  createCopilotFrontendToolActions,
  type MlToolFlowRuntime,
} from "@aifolio/frontend-core/ag-ui";
import {
  handleChangePytorchTargetColumn,
  handleChangeTensorflowTargetColumn,
  handleRandomizePytorchFormFields,
  handleRandomizeTensorflowFormFields,
  handleSetPytorchFormFields,
  handleSetTensorflowFormFields,
  handleStartTensorflowTrainingRuns,
  handleStartPytorchTrainingRuns,
  handleTrainTensorflowModel,
  handleTrainPytorchModel,
} from "@/features/ag-ui-chat/logic/frontendTools.adapter";
import {
  ensurePytorchTab,
  ensureTensorflowTab,
  waitForPytorchForm,
  waitForTensorflowForm,
} from "@/features/ag-ui-chat/logic/copilotFrontendToolsFlow.logic";
import type {
  MlFormPatch,
  MlFormRandomizeArgs,
  MlFrameworkTab,
  PytorchFormPatch,
  TensorflowFormPatch,
} from "@aifolio/contracts/entities/ml-training";

const resolvePytorchFormPatchFromToolArgs = (args: Record<string, unknown>) =>
  resolveMlFormPatchFromToolArgs<PytorchFormPatch>(args);
const resolveTensorflowFormPatchFromToolArgs = (args: Record<string, unknown>) =>
  resolveMlFormPatchFromToolArgs<TensorflowFormPatch>(args);

function defaultNextFrame(): Promise<void> {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function defaultDelay(ms: number): Promise<void> {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

type FrameworkActionBundle<TTrainArgs, TRandomizeArgs> = {
  startTrainingRuns: () => Promise<string>;
  trainModel: (args: TTrainArgs) => Promise<string>;
  setFormFields: (args: Record<string, unknown>) => Promise<string>;
  changeTargetColumn: (args: { target_column?: string; mode?: "different" | "random" | "next" }) => Promise<string>;
  randomizeFormFields: (args: TRandomizeArgs) => Promise<string>;
};

function withDisabledFlag<T extends { name: string }>(
  action: T,
  enabled: boolean
): T & { disabled: boolean } {
  return {
    ...action,
    disabled: !enabled,
  };
}

/**
 * Resolves the currently active ML framework tab from the shared AG-UI store.
 *
 * This reads Zustand state at call time so multi-tool turns do not depend on a
 * stale React closure when a prior tool just switched tabs.
 */
function getCurrentMlFrameworkTab(): MlFrameworkTab | null {
  const { activeTab } = useAgUiWorkspaceStore.getState();
  return activeTab === "pytorch" || activeTab === "tensorflow" ? activeTab : null;
}

function useRegisterFrameworkCopilotActions(
  actions: {
    startTrainingRuns: unknown;
    trainModel: unknown;
    setFormFields: unknown;
    changeTargetColumn: unknown;
    randomizeFormFields: unknown;
  },
  deps: readonly unknown[],
  enabled: boolean
): void {
  useCopilotAction(
    withDisabledFlag(
      actions.startTrainingRuns as Parameters<typeof useCopilotAction>[0] & { name: string },
      enabled
    ),
    [...deps]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.trainModel as Parameters<typeof useCopilotAction>[0] & { name: string },
      enabled
    ),
    []
  );
  useCopilotAction(
    withDisabledFlag(
      actions.setFormFields as Parameters<typeof useCopilotAction>[0] & { name: string },
      enabled
    ),
    [...deps]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.changeTargetColumn as Parameters<typeof useCopilotAction>[0] & { name: string },
      enabled
    ),
    [...deps]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.randomizeFormFields as Parameters<typeof useCopilotAction>[0] & { name: string },
      enabled
    ),
    [...deps]
  );
}

function getFrameworkLabel(framework: MlFrameworkTab): "PyTorch" | "TensorFlow" {
  return framework === ML_TRAINING_FRAMEWORKS.pytorch.tab ? "PyTorch" : "TensorFlow";
}

/**
 * Registers AG-UI frontend tool calls in a hook (view stays thin).
 */
export function useCopilotFrontendTools(runtime: MlToolFlowRuntime = {}): void {
  const { addChartSpec: addGlobalChartSpec, clearChartSpecs: clearGlobalCharts } =
    useCopilotChartActionsAdapter();
  const { addChartSpec: addAgenticChartSpec, clearChartSpecs: clearAgenticCharts } =
    useAgenticResearchChartActionsAdapter();
  const { activeTab, setActiveTab } = useAgUiWorkspaceStateAdapter();
  const router = useRouter();
  const nextFrame = runtime.nextFrame ?? defaultNextFrame;
  const delay = runtime.delay ?? defaultDelay;
  const isMlTab = activeTab === "pytorch" || activeTab === "tensorflow";
  const isPytorchTab = activeTab === "pytorch";
  const isTensorflowTab = activeTab === "tensorflow";

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
    changeTargetColumn: (args: { target_column?: string; mode?: "different" | "random" | "next" }) => unknown,
    randomizeFormFields: (args: TRandomizeArgs) => unknown
  ): FrameworkActionBundle<TTrainArgs, TRandomizeArgs> {
    const frameworkLabel = getFrameworkLabel(framework);
    return {
      startTrainingRuns: async () => {
        await ensureFrameworkTab(framework);
        const result = await startTrainingRuns();
        return formatStartTrainingRunsToolResult(frameworkLabel, result as Record<string, unknown>);
      },
      trainModel: async (args: TTrainArgs) => {
        const result = await trainModel(args);
        return formatTrainModelToolResult(frameworkLabel, result as Record<string, unknown>);
      },
      setFormFields: async (args: Record<string, unknown>) => {
        const patch = resolvePatch(args);
        await ensureFrameworkTab(framework);
        const result = setFormFields(patch);
        if (result.status === "ok") {
          await (runtime.delay ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))))(0);
          await nextFrame();
        }
        return formatSetFormFieldsToolResult(frameworkLabel, result);
      },
      changeTargetColumn: async (args) => {
        await ensureFrameworkTab(framework);
        const result = await changeTargetColumn(args);
        if (
          result &&
          typeof result === "object" &&
          "status" in result &&
          result.status === "ok"
        ) {
          await (runtime.delay ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))))(0);
          await nextFrame();
        }
        return formatChangeTargetColumnToolResult(frameworkLabel, args.target_column, result as Record<string, unknown>);
      },
      randomizeFormFields: async (args: TRandomizeArgs) => {
        await ensureFrameworkTab(framework);
        const result = await randomizeFormFields(args);
        return formatRandomizeFormFieldsToolResult(frameworkLabel, result as Record<string, unknown>);
      },
    };
  }

  const pytorchHandlers = useMemo(
    () =>
      createFrameworkHandlers<
        TrainPytorchModelArgs,
        RandomizePytorchFormFieldsArgs,
        ReturnType<typeof resolvePytorchFormPatchFromToolArgs>
      >(
        "pytorch",
        resolvePytorchFormPatchFromToolArgs,
        handleStartPytorchTrainingRuns,
        handleTrainPytorchModel,
        handleSetPytorchFormFields,
        handleChangePytorchTargetColumn,
        handleRandomizePytorchFormFields
      ),
    [activeTab, router, setActiveTab, nextFrame, delay]
  );

  const tensorflowHandlers = useMemo(
    () =>
      createFrameworkHandlers<
        TrainTensorflowModelArgs,
        RandomizeTensorflowFormFieldsArgs,
        ReturnType<typeof resolveTensorflowFormPatchFromToolArgs>
      >(
        "tensorflow",
        resolveTensorflowFormPatchFromToolArgs,
        handleStartTensorflowTrainingRuns,
        handleTrainTensorflowModel,
        handleSetTensorflowFormFields,
        handleChangeTensorflowTargetColumn,
        handleRandomizeTensorflowFormFields
      ),
    [activeTab, router, setActiveTab, nextFrame, delay]
  );

  const actions = useMemo(
    () =>
      createCopilotFrontendToolActions({
        handleAddChartSpec: ({ chartSpec, chartSpecs }) => {
          const targetAdd = activeTab === "agentic-research" ? addAgenticChartSpec : addGlobalChartSpec;
          const result = handleAddChartSpec({ chartSpec, chartSpecs }, targetAdd);
          return formatAddChartSpecToolResult(result);
        },
        handleClearCharts: () => {
          if (activeTab === "agentic-research") {
            clearAgenticCharts();
          } else {
            clearGlobalCharts();
          }
          return formatClearChartsToolResult();
        },
        handleNavigateToPage: ({ route }: NavigateToPageArgs) => {
          const result = handleNavigateToPage(route);
          if (result.status === "ok") router.push(result.resolvedRoute);
          return formatNavigateToPageToolResult(result);
        },
        handleSetActiveMlFormFields: async (args: Record<string, unknown>) => {
          const framework = getCurrentMlFrameworkTab();
          if (framework === "pytorch") {
            return pytorchHandlers.setFormFields(args);
          }
          if (framework === "tensorflow") {
            return tensorflowHandlers.setFormFields(args);
          }
          return "Unable to update ML form fields: ACTIVE_ML_TAB_REQUIRED.";
        },
        handleChangeActiveMlTargetColumn: async (args) => {
          const framework = getCurrentMlFrameworkTab();
          if (framework === "pytorch") {
            return pytorchHandlers.changeTargetColumn(args);
          }
          if (framework === "tensorflow") {
            return tensorflowHandlers.changeTargetColumn(args);
          }
          return "Unable to change ML target column: ACTIVE_ML_TAB_REQUIRED.";
        },
        handleRandomizeActiveMlFormFields: async (args) => {
          const framework = getCurrentMlFrameworkTab();
          if (framework === "pytorch") {
            return pytorchHandlers.randomizeFormFields(args);
          }
          if (framework === "tensorflow") {
            return tensorflowHandlers.randomizeFormFields(args);
          }
          return "Unable to randomize ML form fields: ACTIVE_ML_TAB_REQUIRED.";
        },
        handleStartActiveMlTrainingRuns: async () => {
          const framework = getCurrentMlFrameworkTab();
          if (framework === "pytorch") {
            return pytorchHandlers.startTrainingRuns();
          }
          if (framework === "tensorflow") {
            return tensorflowHandlers.startTrainingRuns();
          }
          return "Unable to start ML training runs: ACTIVE_ML_TAB_REQUIRED.";
        },
        handleStartPytorchTrainingRuns: pytorchHandlers.startTrainingRuns,
        handleTrainPytorchModel: pytorchHandlers.trainModel,
        handleSetPytorchFormFields: pytorchHandlers.setFormFields,
        handleChangePytorchTargetColumn: pytorchHandlers.changeTargetColumn,
        handleRandomizePytorchFormFields: pytorchHandlers.randomizeFormFields,
        handleStartTensorflowTrainingRuns: tensorflowHandlers.startTrainingRuns,
        handleTrainTensorflowModel: tensorflowHandlers.trainModel,
        handleSetTensorflowFormFields: tensorflowHandlers.setFormFields,
        handleChangeTensorflowTargetColumn: tensorflowHandlers.changeTargetColumn,
        handleRandomizeTensorflowFormFields: tensorflowHandlers.randomizeFormFields,
      }),
    [
      activeTab,
      addAgenticChartSpec,
      addGlobalChartSpec,
      clearAgenticCharts,
      clearGlobalCharts,
      pytorchHandlers,
      router,
      tensorflowHandlers,
    ]
  );

  useCopilotAction(
    actions.addChartSpec as Parameters<typeof useCopilotAction>[0],
    [activeTab, addAgenticChartSpec, addGlobalChartSpec]
  );

  useCopilotAction(
    actions.clearCharts as Parameters<typeof useCopilotAction>[0],
    [activeTab, clearAgenticCharts, clearGlobalCharts]
  );

  useCopilotAction(
    withDisabledFlag(
      actions.setActiveMlFormFields as Parameters<typeof useCopilotAction>[0] & { name: string },
      isMlTab
    ),
    [activeTab, router, setActiveTab]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.changeActiveMlTargetColumn as Parameters<typeof useCopilotAction>[0] & { name: string },
      isMlTab
    ),
    [activeTab, router, setActiveTab]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.randomizeActiveMlFormFields as Parameters<typeof useCopilotAction>[0] & { name: string },
      isMlTab
    ),
    [activeTab, router, setActiveTab]
  );
  useCopilotAction(
    withDisabledFlag(
      actions.startActiveMlTrainingRuns as Parameters<typeof useCopilotAction>[0] & { name: string },
      isMlTab
    ),
    [activeTab, router, setActiveTab]
  );

  // Disabled for /ag-ui by request: keep tool defined but do not register it.
  // useCopilotAction(
  //   actions.navigateToPage as Parameters<typeof useCopilotAction>[0],
  //   [router]
  // );

  const frameworkActionDeps = [activeTab, router, setActiveTab] as const;
  useRegisterFrameworkCopilotActions(
    {
      startTrainingRuns: actions.startPytorchTrainingRuns,
      trainModel: actions.trainPytorchModel,
      setFormFields: actions.setPytorchFormFields,
      changeTargetColumn: actions.changePytorchTargetColumn,
      randomizeFormFields: actions.randomizePytorchFormFields,
    },
    frameworkActionDeps,
    isPytorchTab
  );
  useRegisterFrameworkCopilotActions(
    {
      startTrainingRuns: actions.startTensorflowTrainingRuns,
      trainModel: actions.trainTensorflowModel,
      setFormFields: actions.setTensorflowFormFields,
      changeTargetColumn: actions.changeTensorflowTargetColumn,
      randomizeFormFields: actions.randomizeTensorflowFormFields,
    },
    frameworkActionDeps,
    isTensorflowTab
  );
}
