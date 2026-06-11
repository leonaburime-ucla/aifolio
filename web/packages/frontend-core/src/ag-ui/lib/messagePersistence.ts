export function toPersistableMessages(messages: unknown[]): unknown[] {
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
    if (!Array.isArray(parsed)) return [];

    const normalized = parsed
      .map((entry) => {
        if (!entry || typeof entry !== "object") return null;
        const message = entry as Record<string, unknown>;
        const id = typeof message.id === "string" ? message.id : "";
        const type = "TextMessage";
        const roleRaw = typeof message.role === "string" ? message.role.toLowerCase() : "";
        const role = roleRaw === "user" ? "user" : "assistant";
        const rawContent = message.content;
        const content =
          typeof rawContent === "string"
            ? rawContent
            : Array.isArray(rawContent)
              ? rawContent
                  .map((part) => {
                    if (typeof part === "string") return part;
                    if (!part || typeof part !== "object") return "";
                    const partRecord = part as Record<string, unknown>;
                    if (typeof partRecord.text === "string") return partRecord.text;
                    if (typeof partRecord.content === "string") return partRecord.content;
                    return "";
                  })
                  .filter(Boolean)
                  .join("\n")
              : "";

        if (!id || !content.trim()) return null;
        if (id.startsWith("coagent-state-render-")) return null;
        return { id, type, role, content };
      })
      .filter((entry) => entry !== null);

    const dedupedById = new Map<string, Record<string, unknown>>();
    for (const entry of normalized) {
      const id = String(entry.id ?? "");
      if (!id) continue;
      dedupedById.set(id, entry);
    }
    return Array.from(dedupedById.values());
  } catch {
    return [];
  }
}

export function safeSerialize(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

export function shouldHydratePersistedMessages({
  livePersistableCount,
  liveUserMessageCount,
  persistedCount,
}: {
  livePersistableCount: number;
  liveUserMessageCount: number;
  persistedCount: number;
}): boolean {
  if (persistedCount === 0) return false;
  if (liveUserMessageCount === 0) return true;
  return livePersistableCount === 0;
}

export function shouldSkipEmptyPersistableSync({
  livePersistableCount,
  persistedCount,
}: {
  livePersistableCount: number;
  persistedCount: number;
}): boolean {
  return livePersistableCount === 0 && persistedCount > 0;
}
