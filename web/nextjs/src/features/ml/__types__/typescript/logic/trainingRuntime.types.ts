/**
 * Shared runtime dependency shape for ML training hook logic.
 *
 * Framework-specific runtime types (PyTorch/TensorFlow) are compatible with this base.
 */
export type BaseTrainingRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

