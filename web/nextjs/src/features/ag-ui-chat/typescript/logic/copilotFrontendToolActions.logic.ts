import type {
  ChangePytorchTargetColumnArgs,
  ChangeTensorflowTargetColumnArgs,
  CopilotActionParameter,
  NavigateToPageArgs,
  RandomizePytorchFormFieldsArgs,
  RandomizeTensorflowFormFieldsArgs,
  TrainTensorflowModelArgs,
  TrainPytorchModelArgs,
} from "@/features/ag-ui-chat/__types__/typescript/react/views/copilotTools.types";
import {
  ADD_CHART_SPEC_TOOL,
  CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL,
  CHANGE_PYTORCH_TARGET_COLUMN_TOOL,
  CHANGE_TENSORFLOW_TARGET_COLUMN_TOOL,
  CLEAR_CHARTS_TOOL,
  NAVIGATE_TO_PAGE_TOOL,
  RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL,
  RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL,
  RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL,
  SET_ACTIVE_ML_FORM_FIELDS_TOOL,
  SET_PYTORCH_FORM_FIELDS_TOOL,
  SET_TENSORFLOW_FORM_FIELDS_TOOL,
  START_ACTIVE_ML_TRAINING_RUNS_TOOL,
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
  clearCharts: CopilotToolAction<void, string>;
  navigateToPage: CopilotToolAction<NavigateToPageArgs>;
  setActiveMlFormFields: CopilotToolAction<Record<string, unknown>>;
  changeActiveMlTargetColumn: CopilotToolAction<ChangePytorchTargetColumnArgs>;
  randomizeActiveMlFormFields: CopilotToolAction<RandomizePytorchFormFieldsArgs>;
  startActiveMlTrainingRuns: CopilotToolAction<void>;
  startPytorchTrainingRuns: CopilotToolAction<void>;
  trainPytorchModel: CopilotToolAction<TrainPytorchModelArgs>;
  setPytorchFormFields: CopilotToolAction<Record<string, unknown>>;
  changePytorchTargetColumn: CopilotToolAction<ChangePytorchTargetColumnArgs>;
  randomizePytorchFormFields: CopilotToolAction<RandomizePytorchFormFieldsArgs>;
  startTensorflowTrainingRuns: CopilotToolAction<void>;
  trainTensorflowModel: CopilotToolAction<TrainTensorflowModelArgs>;
  setTensorflowFormFields: CopilotToolAction<Record<string, unknown>>;
  changeTensorflowTargetColumn: CopilotToolAction<ChangeTensorflowTargetColumnArgs>;
  randomizeTensorflowFormFields: CopilotToolAction<RandomizeTensorflowFormFieldsArgs>;
};

type CreateCopilotFrontendToolActionsArgs = {
  handleAddChartSpec: (args: { chartSpec?: unknown; chartSpecs?: unknown[] }) => string;
  handleClearCharts: () => string;
  handleNavigateToPage: (args: NavigateToPageArgs) => string;
  handleSetActiveMlFormFields: (args: Record<string, unknown>) => Promise<string>;
  handleChangeActiveMlTargetColumn: (args: ChangePytorchTargetColumnArgs) => Promise<string>;
  handleRandomizeActiveMlFormFields: (args: RandomizePytorchFormFieldsArgs) => Promise<string>;
  handleStartActiveMlTrainingRuns: () => Promise<string>;
  handleStartPytorchTrainingRuns: () => Promise<string>;
  handleTrainPytorchModel: (args: TrainPytorchModelArgs) => Promise<string>;
  handleSetPytorchFormFields: (args: Record<string, unknown>) => Promise<string>;
  handleChangePytorchTargetColumn: (args: ChangePytorchTargetColumnArgs) => Promise<string>;
  handleRandomizePytorchFormFields: (args: RandomizePytorchFormFieldsArgs) => Promise<string>;
  handleStartTensorflowTrainingRuns: () => Promise<string>;
  handleTrainTensorflowModel: (args: TrainTensorflowModelArgs) => Promise<string>;
  handleSetTensorflowFormFields: (args: Record<string, unknown>) => Promise<string>;
  handleChangeTensorflowTargetColumn: (args: ChangeTensorflowTargetColumnArgs) => Promise<string>;
  handleRandomizeTensorflowFormFields: (args: RandomizeTensorflowFormFieldsArgs) => Promise<string>;
};

const ADD_CHART_SPEC_PARAMETERS: CopilotActionParameter[] = [
  { name: "chartSpec", type: "object", required: false, description: "A single chart spec object to normalize and render." },
  { name: "chartSpecs", type: "object[]", required: false, description: "An array of chart spec objects to normalize and render." },
];

const NAVIGATE_TO_PAGE_PARAMETERS: CopilotActionParameter[] = [
  { name: "route", type: "string", required: true, description: "Route path or alias, such as '/', 'ag-ui', 'agentic research', or '/ml/pytorch'." },
];

const CHANGE_TARGET_COLUMN_PARAMETERS: CopilotActionParameter[] = [
  { name: "target_column", type: "string", required: false, description: "Explicit target column to select. Use when the user names the desired target." },
  { name: "mode", type: "string", required: false, description: "Selection mode when no explicit target is given: different, random, or next. Defaults to different." },
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
  handleSetActiveMlFormFields,
  handleChangeActiveMlTargetColumn,
  handleRandomizeActiveMlFormFields,
  handleStartActiveMlTrainingRuns,
  handleStartPytorchTrainingRuns,
  handleTrainPytorchModel,
  handleSetPytorchFormFields,
  handleChangePytorchTargetColumn,
  handleRandomizePytorchFormFields,
  handleStartTensorflowTrainingRuns,
  handleTrainTensorflowModel,
  handleSetTensorflowFormFields,
  handleChangeTensorflowTargetColumn,
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
    setActiveMlFormFields: {
      name: SET_ACTIVE_ML_FORM_FIELDS_TOOL,
      description:
        "Patch form fields on the currently active ML tab. Use this by default for generic ML prompts when the user does not explicitly name PyTorch or TensorFlow.",
      parameters: ML_TRAINING_FRAMEWORKS.pytorch.setFieldsParameters,
      handler: handleSetActiveMlFormFields,
    },
    changeActiveMlTargetColumn: {
      name: CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL,
      description:
        "Change the target column on the currently active ML tab. Use this by default when the user does not explicitly name PyTorch or TensorFlow.",
      parameters: CHANGE_TARGET_COLUMN_PARAMETERS,
      handler: handleChangeActiveMlTargetColumn,
    },
    randomizeActiveMlFormFields: {
      name: RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL,
      description:
        "Randomize form fields on the currently active ML tab. Use this by default for generic ML prompts when the user does not explicitly name PyTorch or TensorFlow.",
      parameters: ML_TRAINING_FRAMEWORKS.pytorch.randomizeParameters,
      handler: handleRandomizeActiveMlFormFields,
    },
    startActiveMlTrainingRuns: {
      name: START_ACTIVE_ML_TRAINING_RUNS_TOOL,
      description:
        "Start training runs on the currently active ML tab. Use this by default for generic 'train model' prompts when the user does not explicitly name PyTorch or TensorFlow.",
      parameters: [],
      handler: handleStartActiveMlTrainingRuns,
    },
    startPytorchTrainingRuns: pytorchActions.startTrainingRuns,
    trainPytorchModel: pytorchActions.trainModel,
    setPytorchFormFields: pytorchActions.setFormFields,
    changePytorchTargetColumn: {
      name: CHANGE_PYTORCH_TARGET_COLUMN_TOOL,
      description:
        "Change the PyTorch target column. Use this only when the user explicitly names PyTorch; otherwise prefer the active-ML target-column tool.",
      parameters: CHANGE_TARGET_COLUMN_PARAMETERS,
      handler: handleChangePytorchTargetColumn,
    },
    randomizePytorchFormFields: pytorchActions.randomizeFormFields,
    startTensorflowTrainingRuns: tensorflowActions.startTrainingRuns,
    trainTensorflowModel: tensorflowActions.trainModel,
    setTensorflowFormFields: tensorflowActions.setFormFields,
    changeTensorflowTargetColumn: {
      name: CHANGE_TENSORFLOW_TARGET_COLUMN_TOOL,
      description:
        "Change the TensorFlow target column. Use this only when the user explicitly names TensorFlow; otherwise prefer the active-ML target-column tool.",
      parameters: CHANGE_TARGET_COLUMN_PARAMETERS,
      handler: handleChangeTensorflowTargetColumn,
    },
    randomizeTensorflowFormFields: tensorflowActions.randomizeFormFields,
  };
}
