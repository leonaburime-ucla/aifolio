"use client";

import { useCopilotFrontendTools } from "@/features/ag-ui-chat/api/hooks/useCopilotFrontendTools.hooks";

/**
 * Purpose: Register global Copilot frontend tools for charting/navigation/training.
 */
export default function CopilotFrontendTools() {
  useCopilotFrontendTools();
  return null;
}
