import { useShallow } from "zustand/react/shallow";
import type { ChatStatePort } from "@aifolio/contracts/entities/chat";
import { useLandingChatStore } from "@/ui/screens/LandingPage/chat/state/zustand/landingChatStore";

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
