/**
 * Utilities for sanitizing and serializing Copilot message history for persistence.
 */

/**
 * Convert messages to a JSON-serializable format by stripping functions,
 * symbols, bigints, and circular references.
 *
 * @param messages - Raw Copilot messages from runtime context.
 * @returns Safe JSON-like message array that can be persisted to storage.
 */
export function toPersistableMessages(messages: unknown[]): unknown[] {
  const seen = new WeakSet<object>();
  /**
   * JSON replacer that drops unsupported values and circular references.
   *
   * @param _key - Current JSON key (unused).
   * @param value - Current JSON value.
   * @returns Sanitized value or `undefined` to omit field.
   */
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

    // Persist only stable text messages. Dropping tool/internal status payloads
    // avoids restoring transient in-flight runs that trigger terminal-event errors.
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
      .filter((entry): entry is Record<string, unknown> => entry !== null);

    // Keep one record per message id to avoid persistence churn from
    // streaming/status snapshots of the same logical message.
    const dedupedById = new Map<string, Record<string, unknown>>();
    for (const entry of normalized) {
      const id = String(entry.id ?? "");
      if (!id) continue;
      dedupedById.set(id, entry);
    }
    return Array.from(dedupedById.values());
  } catch (error) {
    console.warn("[copilot-message-persistence] serialize_failed", {
      error: String(error),
      incomingCount: messages.length,
    });
    return [];
  }
}

/**
 * Safely serializes a value to JSON.
 *
 * @param value - Any value to serialize.
 * @returns JSON string, or empty string when serialization fails.
 */
export function safeSerialize(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}
