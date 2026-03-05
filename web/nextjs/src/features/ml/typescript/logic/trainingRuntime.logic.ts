import { toast } from "react-hot-toast";
import type { BaseTrainingRuntimeDeps } from "@/features/ml/__types__/typescript/logic/trainingRuntime.types";

/**
 * Builds default runtime dependencies used by ML training hooks.
 *
 * @returns Runtime adapter that provides notifications, scheduling, and clipboard writes.
 */
export function createDefaultTrainingRuntime(): BaseTrainingRuntimeDeps {
  return {
    notifySuccess: (message) => toast.success(message),
    notifyError: (message) => toast.error(message),
    schedule: (callback, delayMs) => {
      setTimeout(callback, delayMs);
    },
    writeClipboardText: async (text) => {
      const clipboard = globalThis.navigator?.clipboard;
      if (!clipboard || typeof clipboard.writeText !== "function") {
        throw new Error("Clipboard API unavailable.");
      }
      await clipboard.writeText(text);
    },
  };
}

