"use client";

import { useEffect, useRef } from "react";
import { useCopilotChatInternal } from "@copilotkit/react-core";
import { useCopilotMessageStateAdapter } from "@/features/copilot-chat/state/adapters/copilotMessageState.adapter";
import {
  toPersistableMessages,
  safeSerialize,
} from "@/features/copilot-chat/utils/messagePersistence.util";

/**
 * Orchestrator hook for Copilot message persistence.
 *
 * Strategy:
 * - On first mount, hydrate Copilot context from persisted store when empty.
 * - Then mirror every message update back into the persisted store.
 *
 * Returns null - this orchestrator is side-effect only.
 */
export function useCopilotMessagePersistenceOrchestrator(): null {
  const { messages, setMessages } = useCopilotChatInternal();
  const { messages: persistedMessages, hasHydrated, setMessages: setPersistedMessages } =
    useCopilotMessageStateAdapter();

  const initializedRef = useRef(false);
  const hydratingFromStoreRef = useRef(false);

  // Debug logging effect
  useEffect(() => {
    console.log("[copilot-message-persistence] status", {
      hasHydrated,
      initialized: initializedRef.current,
      hydratingFromStore: hydratingFromStoreRef.current,
      persistedCount: persistedMessages.length,
      liveCount: messages.length,
    });
  }, [hasHydrated, messages.length, persistedMessages.length]);

  // Hydration effect: restore messages from persisted store on mount
  useEffect(() => {
    if (initializedRef.current) return;
    if (!hasHydrated) {
      console.log("[copilot-message-persistence] waiting_for_store_hydration");
      return;
    }

    initializedRef.current = true;

    if (persistedMessages.length === 0) {
      console.log("[copilot-message-persistence] initialized_without_persisted_messages");
      return;
    }

    hydratingFromStoreRef.current = true;
    setMessages(persistedMessages as never[]);
    console.log("[copilot-message-persistence] hydrated", {
      count: persistedMessages.length,
    });
  }, [hasHydrated, persistedMessages, setMessages]);

  // Sync effect: persist messages to store on changes
  useEffect(() => {
    if (!initializedRef.current) return;
    if (!hasHydrated) return;

    // Avoid clobbering persisted history with transient empty state during hydration.
    if (hydratingFromStoreRef.current) {
      if (messages.length === 0) return;
      hydratingFromStoreRef.current = false;
      console.log("[copilot-message-persistence] hydration_complete", {
        liveCount: messages.length,
      });
    }

    if (messages.length === 0 && persistedMessages.length > 0) {
      console.log("[copilot-message-persistence] skip_sync_empty_live_messages", {
        persistedCount: persistedMessages.length,
      });
      return;
    }

    const persistableMessages = toPersistableMessages(messages as unknown[]);
    const nextSerialized = safeSerialize(persistableMessages);
    const currentSerialized = safeSerialize(persistedMessages);
    if (nextSerialized === currentSerialized) {
      console.log("[copilot-message-persistence] skip_sync_unchanged");
      return;
    }

    setPersistedMessages(persistableMessages);
    console.log("[copilot-message-persistence] synced", {
      liveCount: messages.length,
      persistedCount: persistableMessages.length,
    });
  }, [hasHydrated, messages, persistedMessages, setPersistedMessages]);

  return null;
}
