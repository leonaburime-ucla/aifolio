"use client";

import type { ScreenFeedback } from "@/features/ai-chat/__types__/typescript/uiFeedback.types";

type UIFeedbackProps = {
  feedback: ScreenFeedback | null;
  className?: string;
  onDismiss?: () => void;
  onAction?: () => void;
};

const FEEDBACK_STYLES: Record<ScreenFeedback["kind"], string> = {
  error: "border-red-200 bg-red-50 text-red-700",
  warning: "border-amber-200 bg-amber-50 text-amber-800",
  info: "border-sky-200 bg-sky-50 text-sky-800",
};

const FEEDBACK_LABELS: Record<ScreenFeedback["kind"], string> = {
  error: "Error",
  warning: "Warning",
  info: "Info",
};

/**
 * Feature-local inline feedback surface for persistent chat messages.
 */
export default function UIFeedback({
  feedback,
  className = "",
  onDismiss,
  onAction,
}: UIFeedbackProps) {
  if (!feedback) return null;

  return (
    <div
      className={`flex items-start gap-3 rounded-md border px-3 py-2 text-sm ${FEEDBACK_STYLES[feedback.kind]} ${className}`.trim()}
      role={feedback.kind === "error" ? "alert" : "status"}
    >
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{FEEDBACK_LABELS[feedback.kind]}</p>
        <p className="mt-1 break-words">{feedback.message}</p>
      </div>
      <div className="flex items-center gap-2">
        {feedback.actionLabel && onAction ? (
          <button
            type="button"
            onClick={onAction}
            className="rounded-md border border-current/20 bg-white/70 px-2 py-1 text-xs font-medium"
          >
            {feedback.actionLabel}
          </button>
        ) : null}
        {onDismiss ? (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md border border-current/20 bg-white/70 px-2 py-1 text-xs font-medium"
            aria-label="Dismiss feedback"
          >
            Dismiss
          </button>
        ) : null}
      </div>
    </div>
  );
}
