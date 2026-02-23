/**
 * Utility functions for Copilot message persistence serialization.
 */

/**
 * Convert messages to a JSON-serializable format by stripping functions,
 * symbols, bigints, and circular references.
 */
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
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn("[copilot-message-persistence] serialize_failed", {
      error: String(error),
      incomingCount: messages.length,
    });
    return [];
  }
}

/**
 * Safely serialize a value to JSON string, returning empty string on failure.
 */
export function safeSerialize(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}
