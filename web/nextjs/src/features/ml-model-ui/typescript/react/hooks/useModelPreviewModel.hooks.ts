import { useMemo } from "react";
import {
  buildModelPreviewBullets,
  buildModelPreviewGraph,
} from "@/features/ml-model-ui/typescript/utils/modelPreview.util";
import type {
  ModelPreviewFramework,
  ModelPreviewMode,
  ModelPreviewModel,
} from "@/features/ml-model-ui/__types__/typescript/react/views/modelPreviewModal.types";

/**
 * Builds graph and explainer content for the model preview modal.
 * @param params - Required parameters.
 * @param params.framework - Active ML framework.
 * @param params.mode - Active training mode.
 * @returns Computed graph and explanatory content.
 */
export function useModelPreviewModel({
  framework,
  mode,
}: {
  framework: ModelPreviewFramework;
  mode: ModelPreviewMode;
}): ModelPreviewModel {
  return useMemo(() => {
    const rawGraph = buildModelPreviewGraph({ framework, mode });
    const data = buildModelPreviewBullets({ mode });

    const graph = {
      ...rawGraph,
      nodes: rawGraph.nodes.map((node) => ({
        ...node,
        style: { ...node.style, color: "#18181b" },
      })),
    };

    return { graph, data };
  }, [framework, mode]);
}
