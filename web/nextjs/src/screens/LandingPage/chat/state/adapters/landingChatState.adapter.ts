import { useShallow } from "zustand/react/shallow";
import { useLandingChatStore } from "@/screens/LandingPage/chat/state/zustand/landingChatStore";
import type { ChatStatePort } from "@/features/ai-chat/__types__/typescript/chat.types";

/**
 * Adapter that exposes landing chat store via a neutral state port.
 */
export function useLandingChatStateAdapter(): ChatStatePort {
  const state = useLandingChatStore(
    useShallow((store) => ({
      messages: store.messages,
      inputHistory: store.inputHistory,
      historyCursor: store.historyCursor,
      isSending: store.isSending,
      modelOptions: store.modelOptions,
      selectedModelId: store.selectedModelId,
      isModelsLoading: store.isModelsLoading,
      screenFeedback: store.screenFeedback,
    }))
  );

  const actions = useLandingChatStore(
    useShallow((store) => ({
      addMessage: store.addMessage,
      addInputToHistory: store.addInputToHistory,
      moveHistoryCursor: store.moveHistoryCursor,
      resetHistoryCursor: store.resetHistoryCursor,
      setSending: store.setSending,
      setModelOptions: store.setModelOptions,
      setSelectedModelId: store.setSelectedModelId,
      setModelsLoading: store.setModelsLoading,
      setScreenFeedback: store.setScreenFeedback,
    }))
  );

  return { state, actions };
}
