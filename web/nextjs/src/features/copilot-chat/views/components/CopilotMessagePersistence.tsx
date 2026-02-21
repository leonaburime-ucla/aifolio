"use client";

import { useEffect, useRef } from "react";
import { useCopilotChatInternal } from "@copilotkit/react-core";
import { useCopilotMessageStore } from "@/features/copilot-chat/state/zustand/copilotMessageStore";

function toPersistableMessages(messages: unknown[]): unknown[] {
  const seen = new WeakSet<object>();
  const replacer = (_key: string, value: unknown) => {
    if (typeof value === "function" || typeof value === "symbol") return undefined;
    if (typeof value === "bigint") return String(value);
    if (value && typeof value === "object") {
      if (seen.has(value as object)) return undefined;
      seen.add(value as object);
    }
    return value;
  };

  try {
    const serialized = JSON.stringify(messages, replacer);
    if (!serialized) return [];
    const parsed = JSON.parse(serialized);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn("[copilot-message-persistence] serialize_failed", {
      error: String(error),
      incomingCount: messages.length,
    });
    return [];
  }
}

function safeSerialize(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

/**
 * Persists Copilot messages across page navigations/remounts.
 *
 * Strategy:
 * - On first mount, hydrate Copilot context from persisted store when empty.
 * - Then mirror every message update back into the persisted store.
 */
export default function CopilotMessagePersistence() {
  const { messages, setMessages } = useCopilotChatInternal();
  const persistedMessages = useCopilotMessageStore((state) => state.messages);
  const hasHydrated = useCopilotMessageStore((state) => state.hasHydrated);
  const setPersistedMessages = useCopilotMessageStore((state) => state.setMessages);
  const initializedRef = useRef(false);
  const hydratingFromStoreRef = useRef(false);

  useEffect(() => {
    console.log("[copilot-message-persistence] status", {
      hasHydrated,
      initialized: initializedRef.current,
      hydratingFromStore: hydratingFromStoreRef.current,
      persistedCount: persistedMessages.length,
      liveCount: messages.length,
    });
  }, [hasHydrated, messages.length, persistedMessages.length]);

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
