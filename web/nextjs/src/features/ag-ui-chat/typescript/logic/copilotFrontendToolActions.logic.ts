import type {
  CopilotActionParameter,
  NavigateToPageArgs,
  RandomizePytorchFormFieldsArgs,
  RandomizeTensorflowFormFieldsArgs,
  TrainTensorflowModelArgs,
  TrainPytorchModelArgs,
} from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotTools.types";
import {
  ADD_CHART_SPEC_TOOL,
  CLEAR_CHARTS_TOOL,
  NAVIGATE_TO_PAGE_TOOL,
  RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL,
  RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL,
  SET_PYTORCH_FORM_FIELDS_TOOL,
  SET_TENSORFLOW_FORM_FIELDS_TOOL,
  START_PYTORCH_TRAINING_RUNS_TOOL,
  START_TENSORFLOW_TRAINING_RUNS_TOOL,
  TRAIN_TENSORFLOW_MODEL_TOOL,
  TRAIN_PYTORCH_MODEL_TOOL,
} from "@/features/ag-ui-chat/typescript/config/frontendTools.config";
import {
  createMlFrameworkActions,
  type CopilotToolAction,
} from "@/features/ml/typescript/ai/agUi/mlTrainingToolActions.logic";
import { ML_TRAINING_FRAMEWORKS } from "@/features/ml/typescript/ai/agUi/mlTrainingFrameworkMetadata.logic";

/**
 * Copilot frontend tool action-definition factory.
 *
 * Purpose:
 * - Build stable action config objects outside React hook bodies.
 * - Centralize names/descriptions/parameters and injected handlers.
 * - Keep `useCopilotFrontendTools` focused on registration wiring.
 */

export type CopilotFrontendToolActions = {
  addChartSpec: CopilotToolAction<{ chartSpec?: unknown; chartSpecs?: unknown[] }>;
  clearCharts: CopilotToolAction<void, { status: "ok"; cleared: true }>;
  navigateToPage: CopilotToolAction<NavigateToPageArgs>;
  startPytorchTrainingRuns: CopilotToolAction<void>;
  trainPytorchModel: CopilotToolAction<TrainPytorchModelArgs>;
  setPytorchFormFields: CopilotToolAction<Record<string, unknown>>;
  randomizePytorchFormFields: CopilotToolAction<RandomizePytorchFormFieldsArgs>;
  startTensorflowTrainingRuns: CopilotToolAction<void>;
  trainTensorflowModel: CopilotToolAction<TrainTensorflowModelArgs>;
  setTensorflowFormFields: CopilotToolAction<Record<string, unknown>>;
  randomizeTensorflowFormFields: CopilotToolAction<RandomizeTensorflowFormFieldsArgs>;
};

type CreateCopilotFrontendToolActionsArgs = {
  handleAddChartSpec: (args: { chartSpec?: unknown; chartSpecs?: unknown[] }) => unknown;
  handleClearCharts: () => { status: "ok"; cleared: true };
  handleNavigateToPage: (args: NavigateToPageArgs) => unknown;
  handleStartPytorchTrainingRuns: () => Promise<unknown>;
  handleTrainPytorchModel: (args: TrainPytorchModelArgs) => Promise<unknown>;
  handleSetPytorchFormFields: (args: Record<string, unknown>) => Promise<unknown>;
  handleRandomizePytorchFormFields: (args: RandomizePytorchFormFieldsArgs) => Promise<unknown>;
  handleStartTensorflowTrainingRuns: () => Promise<unknown>;
  handleTrainTensorflowModel: (args: TrainTensorflowModelArgs) => Promise<unknown>;
  handleSetTensorflowFormFields: (args: Record<string, unknown>) => Promise<unknown>;
  handleRandomizeTensorflowFormFields: (args: RandomizeTensorflowFormFieldsArgs) => Promise<unknown>;
};

const ADD_CHART_SPEC_PARAMETERS: CopilotActionParameter[] = [
  { name: "chartSpec", type: "object", required: false, description: "A single chart spec object to normalize and render." },
  { name: "chartSpecs", type: "object[]", required: false, description: "An array of chart spec objects to normalize and render." },
];

const NAVIGATE_TO_PAGE_PARAMETERS: CopilotActionParameter[] = [
  { name: "route", type: "string", required: true, description: "Route path or alias, such as '/', 'ag-ui', 'agentic research', or '/ml/pytorch'." },
];

/**
 * Creates Copilot action definitions from injected handlers.
 *
 * @param args Injected behavior handlers for each action.
 * @returns Structured action definitions for registration.
 */
export function createCopilotFrontendToolActions({
  handleAddChartSpec,
  handleClearCharts,
  handleNavigateToPage,
  handleStartPytorchTrainingRuns,
  handleTrainPytorchModel,
  handleSetPytorchFormFields,
  handleRandomizePytorchFormFields,
  handleStartTensorflowTrainingRuns,
  handleTrainTensorflowModel,
  handleSetTensorflowFormFields,
  handleRandomizeTensorflowFormFields,
}: CreateCopilotFrontendToolActionsArgs): CopilotFrontendToolActions {
  const pytorchActions = createMlFrameworkActions<
    TrainPytorchModelArgs,
    RandomizePytorchFormFieldsArgs
  >({
    frameworkLabel: ML_TRAINING_FRAMEWORKS.pytorch.label,
    startToolName: START_PYTORCH_TRAINING_RUNS_TOOL,
    trainToolName: TRAIN_PYTORCH_MODEL_TOOL,
    setFieldsToolName: SET_PYTORCH_FORM_FIELDS_TOOL,
    randomizeFieldsToolName: RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL,
    trainParams: ML_TRAINING_FRAMEWORKS.pytorch.trainParameters,
    setFieldsParams: ML_TRAINING_FRAMEWORKS.pytorch.setFieldsParameters,
    randomizeParams: ML_TRAINING_FRAMEWORKS.pytorch.randomizeParameters,
    setFieldsDescription: "Set/patch PyTorch training form fields on the /ag-ui PyTorch tab.",
    startHandler: handleStartPytorchTrainingRuns,
    trainHandler: handleTrainPytorchModel,
    setFieldsHandler: handleSetPytorchFormFields,
    randomizeHandler: handleRandomizePytorchFormFields,
  });

  const tensorflowActions = createMlFrameworkActions<
    TrainTensorflowModelArgs,
    RandomizeTensorflowFormFieldsArgs
  >({
    frameworkLabel: ML_TRAINING_FRAMEWORKS.tensorflow.label,
    startToolName: START_TENSORFLOW_TRAINING_RUNS_TOOL,
    trainToolName: TRAIN_TENSORFLOW_MODEL_TOOL,
    setFieldsToolName: SET_TENSORFLOW_FORM_FIELDS_TOOL,
    randomizeFieldsToolName: RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL,
    trainParams: ML_TRAINING_FRAMEWORKS.tensorflow.trainParameters,
    setFieldsParams: ML_TRAINING_FRAMEWORKS.tensorflow.setFieldsParameters,
    randomizeParams: ML_TRAINING_FRAMEWORKS.tensorflow.randomizeParameters,
    setFieldsDescription: "Set/patch TensorFlow training form fields on the /ag-ui TensorFlow tab.",
    startHandler: handleStartTensorflowTrainingRuns,
    trainHandler: handleTrainTensorflowModel,
    setFieldsHandler: handleSetTensorflowFormFields,
    randomizeHandler: handleRandomizeTensorflowFormFields,
  });

  return {
    addChartSpec: {
      name: ADD_CHART_SPEC_TOOL,
      description: "Add one chart spec or an array of chart specs to the frontend chart store for immediate rendering.",
      parameters: ADD_CHART_SPEC_PARAMETERS,
      handler: handleAddChartSpec,
    },
    clearCharts: {
      name: CLEAR_CHARTS_TOOL,
      description: "Clear all chart specs from the frontend chart store.",
      parameters: [],
      handler: handleClearCharts,
    },
    navigateToPage: {
      name: NAVIGATE_TO_PAGE_TOOL,
      description: "Navigate the user to another app page.",
      parameters: NAVIGATE_TO_PAGE_PARAMETERS,
      handler: handleNavigateToPage,
    },
    startPytorchTrainingRuns: pytorchActions.startTrainingRuns,
    trainPytorchModel: pytorchActions.trainModel,
    setPytorchFormFields: pytorchActions.setFormFields,
    randomizePytorchFormFields: pytorchActions.randomizeFormFields,
    startTensorflowTrainingRuns: tensorflowActions.startTrainingRuns,
    trainTensorflowModel: tensorflowActions.trainModel,
    setTensorflowFormFields: tensorflowActions.setFormFields,
    randomizeTensorflowFormFields: tensorflowActions.randomizeFormFields,
  };
}
