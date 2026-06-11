"use client";

import type { ComponentType } from "react";
import type { AgUiWorkspaceTab } from "@aifolio/contracts/entities/ag-ui";
import AgenticResearchAgUiSurface from "@/features/agentic-research/react/views/screens/AgenticResearchAgUiSurface";
import ChartsWorkspaceSurface from "@/features/recharts/react/views/screens/ChartsWorkspaceSurface";
import PytorchAgUiSurface from "@/features/ml/react/views/screens/PytorchAgUiSurface";
import TensorflowAgUiSurface from "@/features/ml/react/views/screens/TensorflowAgUiSurface";

export const AG_UI_WORKSPACE_SURFACES: Record<
  AgUiWorkspaceTab,
  ComponentType
> = {
  charts: ChartsWorkspaceSurface,
  "agentic-research": AgenticResearchAgUiSurface,
  pytorch: PytorchAgUiSurface,
  tensorflow: TensorflowAgUiSurface,
};
