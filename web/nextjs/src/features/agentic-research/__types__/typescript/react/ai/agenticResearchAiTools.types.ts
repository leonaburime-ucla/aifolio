export type CopilotActionParameterType =
  | "string"
  | "number"
  | "boolean"
  | "object"
  | "string[]"
  | "number[]"
  | "boolean[]"
  | "object[]";

export type CopilotActionParameter = {
  name: string;
  type: CopilotActionParameterType;
  required: boolean;
  description: string;
};
