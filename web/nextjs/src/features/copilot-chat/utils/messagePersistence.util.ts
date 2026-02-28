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
