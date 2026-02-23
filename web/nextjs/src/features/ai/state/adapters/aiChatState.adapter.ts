import { useShallow } from "zustand/react/shallow";
import { useAiChatStore } from "@/features/ai/state/zustand/aiChatStore";
import type { ChatStatePort } from "@/features/ai/types/chat.types";

/**
 * Adapter that exposes AI chat store via a neutral state port.
 */
export function useAiChatStateAdapter(): ChatStatePort {
  const state = useAiChatStore(
    useShallow((store) => ({
      messages: store.messages,
      inputHistory: store.inputHistory,
      historyCursor: store.historyCursor,
      isSending: store.isSending,
      modelOptions: store.modelOptions,
      selectedModelId: store.selectedModelId,
      isModelsLoading: store.isModelsLoading,
    }))
  );

  const actions = useAiChatStore(
    useShallow((store) => ({
      addMessage: store.addMessage,
      addInputToHistory: store.addInputToHistory,
      moveHistoryCursor: store.moveHistoryCursor,
      resetHistoryCursor: store.resetHistoryCursor,
      setSending: store.setSending,
      setModelOptions: store.setModelOptions,
      setSelectedModelId: store.setSelectedModelId,
      setModelsLoading: store.setModelsLoading,
    }))
  );

  return { state, actions };
}
