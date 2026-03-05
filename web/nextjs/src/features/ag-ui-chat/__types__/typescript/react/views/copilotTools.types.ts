import type {
  PytorchRandomizeArgs,
  TensorflowRandomizeArgs,
  TrainPytorchModelArgs,
  TrainTensorflowModelArgs,
} from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";

export type CopilotActionParameter = {
  name: string;
  type: string;
  required: boolean;
  description: string;
};

export type AgUiTabSwitchArgs = {
  tab: string;
};

export type NavigateToPageArgs = {
  route: string;
};

export type RandomizePytorchFormFieldsArgs = PytorchRandomizeArgs;
export type RandomizeTensorflowFormFieldsArgs = TensorflowRandomizeArgs;

export type { TrainPytorchModelArgs, TrainTensorflowModelArgs };
