export type CopilotActionParameter = {
  name: string;
  type: string;
  required: boolean;
  description: string;
};

export type CopilotToolAction<TArgs = unknown, TResult = unknown> = {
  name: string;
  description: string;
  parameters: CopilotActionParameter[];
  handler: (args: TArgs) => TResult | Promise<TResult>;
};

type MlFrameworkActionConfig<TTrainArgs, TRandomizeArgs> = {
  frameworkLabel: "PyTorch" | "TensorFlow";
  startToolName: string;
  trainToolName: string;
  setFieldsToolName: string;
  randomizeFieldsToolName: string;
  trainParams: CopilotActionParameter[];
  setFieldsParams: CopilotActionParameter[];
  randomizeParams: CopilotActionParameter[];
  setFieldsDescription: string;
  startHandler: () => Promise<unknown>;
  trainHandler: (args: TTrainArgs) => Promise<unknown>;
  setFieldsHandler: (args: Record<string, unknown>) => Promise<unknown>;
  randomizeHandler: (args: TRandomizeArgs) => Promise<unknown>;
};

/**
 * Creates the shared 4-action training tool set for an ML framework.
 */
export function createMlFrameworkActions<TTrainArgs, TRandomizeArgs>({
  frameworkLabel,
  startToolName,
  trainToolName,
  setFieldsToolName,
  randomizeFieldsToolName,
  trainParams,
  setFieldsParams,
  randomizeParams,
  setFieldsDescription,
  startHandler,
  trainHandler,
  setFieldsHandler,
  randomizeHandler,
}: MlFrameworkActionConfig<TTrainArgs, TRandomizeArgs>) {
  return {
    startTrainingRuns: {
      name: startToolName,
      description:
        `Start ${frameworkLabel} training using the current /ag-ui ${frameworkLabel} form state (selected dataset, target column, sweep/autodistill, and hyperparameters). Use this only when the user explicitly names ${frameworkLabel}; otherwise prefer the active-ML training tool.`,
      parameters: [],
      handler: startHandler,
    } satisfies CopilotToolAction<void>,
    trainModel: {
      name: trainToolName,
      description:
        `Start one backend ${frameworkLabel} training run with explicit dataset_id and target_column arguments. Use this only when the user explicitly names ${frameworkLabel} and those values are explicitly provided.`,
      parameters: trainParams,
      handler: trainHandler,
    } satisfies CopilotToolAction<TTrainArgs>,
    setFormFields: {
      name: setFieldsToolName,
      description: `${setFieldsDescription} Use this only when the user explicitly names ${frameworkLabel}; otherwise prefer the active-ML set-fields tool.`,
      parameters: setFieldsParams,
      handler: setFieldsHandler,
    } satisfies CopilotToolAction<Record<string, unknown>>,
    randomizeFormFields: {
      name: randomizeFieldsToolName,
      description:
        `Randomize ${frameworkLabel} form fields only when the user explicitly asks to randomize and explicitly names ${frameworkLabel}. Otherwise prefer the active-ML randomize tool. Requires confirm_randomize=true.`,
      parameters: randomizeParams,
      handler: randomizeHandler,
    } satisfies CopilotToolAction<TRandomizeArgs>,
  };
}
