"use client";

import { useAgUiModelSelector } from "@/features/ag-ui-chat/typescript/react/hooks/useAgUiModelSelector.hooks";

/**
 * AG-UI model dropdown bound to model selector hook state.
 */
export default function AgUiModelSelector() {
  const { modelOptions, selectedModelId, isModelsLoading, setSelectedModelId } =
    useAgUiModelSelector();

  return (
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
  );
}
