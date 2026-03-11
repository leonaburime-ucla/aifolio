import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";
import {
  ADD_CHART_SPEC_TOOL,
  CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL,
  CHANGE_PYTORCH_TARGET_COLUMN_TOOL,
  CHANGE_TENSORFLOW_TARGET_COLUMN_TOOL,
  CLEAR_CHARTS_TOOL,
  RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL,
  RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL,
  RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL,
  SET_ACTIVE_ML_FORM_FIELDS_TOOL,
  SET_PYTORCH_FORM_FIELDS_TOOL,
  SET_TENSORFLOW_FORM_FIELDS_TOOL,
  START_ACTIVE_ML_TRAINING_RUNS_TOOL,
  START_PYTORCH_TRAINING_RUNS_TOOL,
  START_TENSORFLOW_TRAINING_RUNS_TOOL,
  SWITCH_AG_UI_TAB_TOOL,
  TRAIN_TENSORFLOW_MODEL_TOOL,
  TRAIN_PYTORCH_MODEL_TOOL,
} from "@/features/ag-ui-chat/typescript/config/frontendTools.config";
import {
  AR_ADD_CHART_SPEC_TOOL,
  AR_CLEAR_CHARTS_TOOL,
  AR_REMOVE_CHART_SPEC_TOOL,
  AR_REORDER_CHART_SPECS_TOOL,
  AR_SET_ACTIVE_DATASET_TOOL,
} from "@/features/agentic-research/typescript/config/frontendTools.config";

/**
 * AG-UI workspace tools catalog per tab.
 *
 * Purpose:
 * - Surface available tool calls for the active AG-UI page.
 * - Keep UI metadata centralized and testable.
 */

export type AgUiToolInfo = {
  name: string;
  summary: string;
};

const BASE_TOOLS: AgUiToolInfo[] = [
  { name: SWITCH_AG_UI_TAB_TOOL, summary: "Switches AG-UI workspace tabs." },
];

const CHART_TOOLS: AgUiToolInfo[] = [
  { name: ADD_CHART_SPEC_TOOL, summary: "Adds chart specs to the chart canvas." },
  { name: CLEAR_CHARTS_TOOL, summary: "Clears rendered charts." },
];

const PYTORCH_TOOLS: AgUiToolInfo[] = [
  { name: SET_ACTIVE_ML_FORM_FIELDS_TOOL, summary: "Patches ML form fields on the currently active ML tab." },
  { name: CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL, summary: "Changes the target column on the currently active ML tab." },
  { name: RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL, summary: "Randomizes form fields on the currently active ML tab." },
  { name: START_ACTIVE_ML_TRAINING_RUNS_TOOL, summary: "Starts training runs on the currently active ML tab." },
  { name: SET_PYTORCH_FORM_FIELDS_TOOL, summary: "Patches PyTorch form fields." },
  { name: CHANGE_PYTORCH_TARGET_COLUMN_TOOL, summary: "Changes the PyTorch target column." },
  { name: RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL, summary: "Randomizes PyTorch form fields with various values." },
  { name: START_PYTORCH_TRAINING_RUNS_TOOL, summary: "Starts training runs from current form state." },
  { name: TRAIN_PYTORCH_MODEL_TOOL, summary: "Runs one explicit backend training call." },
];

const TENSORFLOW_TOOLS: AgUiToolInfo[] = [
  { name: SET_ACTIVE_ML_FORM_FIELDS_TOOL, summary: "Patches ML form fields on the currently active ML tab." },
  { name: CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL, summary: "Changes the target column on the currently active ML tab." },
  { name: RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL, summary: "Randomizes form fields on the currently active ML tab." },
  { name: START_ACTIVE_ML_TRAINING_RUNS_TOOL, summary: "Starts training runs on the currently active ML tab." },
  { name: SET_TENSORFLOW_FORM_FIELDS_TOOL, summary: "Patches TensorFlow form fields." },
  { name: CHANGE_TENSORFLOW_TARGET_COLUMN_TOOL, summary: "Changes the TensorFlow target column." },
  { name: RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL, summary: "Randomizes TensorFlow form fields with various values." },
  { name: START_TENSORFLOW_TRAINING_RUNS_TOOL, summary: "Starts training runs from current form state." },
  { name: TRAIN_TENSORFLOW_MODEL_TOOL, summary: "Runs one explicit backend training call." },
];

const AGENTIC_RESEARCH_TOOLS: AgUiToolInfo[] = [
  { name: AR_ADD_CHART_SPEC_TOOL, summary: "Adds Agentic Research chart specs." },
  { name: AR_CLEAR_CHARTS_TOOL, summary: "Clears Agentic Research charts." },
  { name: AR_REMOVE_CHART_SPEC_TOOL, summary: "Removes a chart by id." },
  { name: AR_REORDER_CHART_SPECS_TOOL, summary: "Reorders chart cards." },
  { name: AR_SET_ACTIVE_DATASET_TOOL, summary: "Selects active Agentic Research dataset." },
];

/**
 * Returns tools relevant to the active AG-UI tab.
 *
 * @param tab Active workspace tab.
 * @returns Ordered tool list for the tab.
 */
export function getAgUiToolsForTab(tab: AgUiWorkspaceTab): AgUiToolInfo[] {
  if (tab === "agentic-research") {
    return [...BASE_TOOLS, ...CHART_TOOLS, ...AGENTIC_RESEARCH_TOOLS];
  }
  if (tab === "pytorch") {
    return [...BASE_TOOLS, ...PYTORCH_TOOLS];
  }
  if (tab === "tensorflow") {
    return [...BASE_TOOLS, ...TENSORFLOW_TOOLS];
  }
  return [...BASE_TOOLS, ...CHART_TOOLS];
}
