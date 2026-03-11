"use client";

import { useAgUiModelSelectorOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/agUiModelSelector.orchestrator";

/**
 * AG-UI model dropdown bound to model selector orchestrator state.
 *
 * Displays a backend-unavailable warning banner when the backend
 * cannot be reached, and the model dropdown for selecting AI models.
 */
export default function AgUiModelSelector() {
  const { modelOptions, selectedModelId, isModelsLoading, backendError, setSelectedModelId } =
    useAgUiModelSelectorOrchestrator();

  return (
    <div className="flex flex-col gap-2">
      {backendError ? (
        <div className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          <span aria-hidden="true">⚠</span>
          <span>{backendError}</span>
        </div>
      ) : null}
      <select
        value={selectedModelId ?? ""}
        onChange={(event) => setSelectedModelId(event.target.value || null)}
        disabled={modelOptions.length === 0}
        className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-300 disabled:cursor-not-allowed disabled:bg-zinc-100"
        aria-label="Select AG-UI model"
      >
        {modelOptions.length === 0 ? (
          <option value="">{isModelsLoading ? "Loading models..." : "No models available"}</option>
        ) : (
          modelOptions.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
