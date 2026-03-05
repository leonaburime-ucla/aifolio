import type {
  GroupSklearnToolsInput,
  GroupSklearnToolsResult,
} from "@/features/agentic-research/__types__/typescript/logic/agenticResearchTools.types";

/**
 * Group sklearn tool identifiers into deterministic display categories.
 *
 * @param input - Required sklearn tools grouping input.
 * @returns Grouped tools keyed by category.
 */
export function groupSklearnTools(
  input: GroupSklearnToolsInput
): GroupSklearnToolsResult {
  return input.tools.reduce<GroupSklearnToolsResult>((acc, tool) => {
    let group = "Other";
    if (tool.includes("regression")) group = "Regression";
    else if (tool.includes("classification")) group = "Classification";
    else if (tool.includes("clustering")) group = "Clustering";
    else if (
      tool.includes("pca") ||
      tool.includes("svd") ||
      tool.includes("ica") ||
      tool.includes("nmf") ||
      tool.includes("tsne")
    ) {
      group = "Decomposition & Embeddings";
    } else if (
      tool.includes("scaler") ||
      tool.includes("encoder") ||
      tool.includes("transformer") ||
      tool.includes("imputer")
    ) {
      group = "Preprocessing";
    } else if (tool.includes("select_") || tool.includes("rfe") || tool.includes("rfecv")) {
      group = "Feature Selection";
    } else if (tool.includes("train_test_split")) {
      group = "Model Selection";
    } else if (
      tool.includes("accuracy") ||
      tool.includes("precision") ||
      tool.includes("recall") ||
      tool.includes("f1") ||
      tool.includes("roc") ||
      tool.includes("auc")
    ) {
      group = "Metrics";
    }

    if (!acc[group]) acc[group] = [];
    acc[group].push(tool);
    return acc;
  }, {});
}
