"use client";

import { useAgUiPageQuerySyncOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/agUiPageQuerySync.orchestrator";

/**
 * Applies `/ag-ui?page=<alias>` query param to workspace tab state.
 */
export default function AgUiPageQuerySync() {
  useAgUiPageQuerySyncOrchestrator();
  return null;
}
