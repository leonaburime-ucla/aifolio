"use client";

/**
 * Purpose: Keep Copilot runtime messages and persisted local message history in sync.
 */
import { useEffect, useRef } from "react";
import { useCopilotChatInternal } from "@copilotkit/react-core";
import { useCopilotMessageStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/copilotMessageState.adapter";
import {
  toPersistableMessages,
  safeSerialize,
} from "@/features/ag-ui-chat/typescript/logic/messagePersistence.util";
import {
  shouldHydratePersistedMessages,
  shouldSkipEmptyPersistableSync,
} from "@/features/ag-ui-chat/typescript/logic/copilotMessagePersistence.logic";

const DEBUG_COPILOT =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG !== "0" &&
  process.env.COPILOT_DEBUG !== "0";

function debugLog(event: string, meta?: Record<string, unknown>) {
  if (!DEBUG_COPILOT) return;
  if (meta) {
    console.log(event, meta);
    return;
  }
  console.log(event);
}

function countLiveUserMessages(messages: unknown[]): number {
  return messages.filter((entry) => {
    if (!entry || typeof entry !== "object") return false;
    const role = String((entry as { role?: unknown }).role ?? "").toLowerCase();
    return role === "user";
  }).length;
}

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
    debugLog("[copilot-message-persistence] status", {
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
      debugLog("[copilot-message-persistence] waiting_for_store_hydration");
      return;
    }

    const livePersistable = toPersistableMessages(messages as unknown[]);
    const liveUserMessageCount = countLiveUserMessages(messages as unknown[]);
    const shouldHydrate = shouldHydratePersistedMessages({
      livePersistableCount: livePersistable.length,
      liveUserMessageCount,
      persistedCount: persistedMessages.length,
    });

    // If runtime already contains meaningful/persistable messages, keep live state.
    if (!shouldHydrate) {
      initializedRef.current = true;
      debugLog("[copilot-message-persistence] initialized_with_live_messages", {
        liveCount: messages.length,
        livePersistableCount: livePersistable.length,
        liveUserMessageCount,
      });
      return;
    }

    hydratingFromStoreRef.current = true;
    setMessages(persistedMessages as never[]);
    initializedRef.current = true;
    debugLog("[copilot-message-persistence] hydrated", {
      count: persistedMessages.length,
    });
  }, [hasHydrated, messages, persistedMessages, setMessages]);

  // Sync effect: persist messages to store on changes
  useEffect(() => {
    if (!initializedRef.current) return;
    if (!hasHydrated) return;
    const livePersistableMessages = toPersistableMessages(messages as unknown[]);

    // Avoid clobbering persisted history with transient empty state during hydration.
    if (hydratingFromStoreRef.current) {
      if (livePersistableMessages.length === 0 && persistedMessages.length > 0) {
        // Copilot runtime can emit transient status/system snapshots immediately
        // after mount. Re-apply persisted history until persistable messages stick.
        debugLog("[copilot-message-persistence] hydration_retry_restore", {
          liveCount: messages.length,
          persistedCount: persistedMessages.length,
        });
        setMessages(persistedMessages as never[]);
        return;
      }
      hydratingFromStoreRef.current = false;
      debugLog("[copilot-message-persistence] hydration_complete", {
        liveCount: messages.length,
      });
    }

    if (messages.length === 0) {
      debugLog("[copilot-message-persistence] skip_sync_empty_live_messages", {
        persistedCount: persistedMessages.length,
      });
      return;
    }

    const persistableMessages = livePersistableMessages;
    if (
      shouldSkipEmptyPersistableSync({
        livePersistableCount: persistableMessages.length,
        persistedCount: persistedMessages.length,
      })
    ) {
      debugLog("[copilot-message-persistence] skip_sync_empty_persistable_snapshot", {
        liveCount: messages.length,
        persistedCount: persistedMessages.length,
      });
      return;
    }
    const nextSerialized = safeSerialize(persistableMessages);
    const currentSerialized = safeSerialize(persistedMessages);
    if (nextSerialized === currentSerialized) {
      debugLog("[copilot-message-persistence] skip_sync_unchanged");
      return;
    }

    setPersistedMessages(persistableMessages);
    debugLog("[copilot-message-persistence] synced", {
      liveCount: messages.length,
      persistedCount: persistableMessages.length,
    });
  }, [hasHydrated, messages, persistedMessages, setMessages, setPersistedMessages]);

  return null;
}
