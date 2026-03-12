/**
 * Persistent feedback shown inline within the AI chat surface.
 */
export type ScreenFeedback = {
  kind: "error" | "warning" | "info";
  code: string;
  message: string;
  retryable?: boolean;
  actionLabel?: string;
};

/**
 * Ephemeral feedback intended for transient notifications such as toasts.
 */
export type NotificationFeedback = {
  kind: "success" | "warning" | "info";
  code: string;
  message: string;
  ttlMs?: number;
};
