import { useEffect } from "react";
import type {
  PytorchFormBridge,
  PytorchFormBridgeBindings,
} from "@/features/ml/__types__/typescript/react/ai/tools/pytorchFormBridge.tools.types";
import { applyPytorchBridgePatch } from "@/features/ml/typescript/react/ai/tools/pytorchFormBridgePatch.tools";

/**
 * React hook that binds the global PyTorch tool-call bridge lifecycle to component mount state.
 *
 * Relative to sibling modules in this folder:
 * - `pytorchFormBridgePatch.tools.ts` contains pure patch logic.
 * - This file contains only browser-global registration/cleanup side effects.
 */

declare global {
  interface Window {
    __AIFOLIO_PYTORCH_FORM_BRIDGE__?: PytorchFormBridge;
  }
}

/**
 * Waits one macro-task plus one animation frame so React state updates caused by
 * tool-call patches settle before training is triggered.
 *
 * @returns Promise that resolves when queued UI work has had a chance to flush.
 */
async function flushUiWork(): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(() => resolve(), 0);
  });

  await new Promise<void>((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(() => resolve(), 0);
  });
}

/**
 * Registers a controlled global bridge that AI tool calls can use to patch the
 * PyTorch form and trigger training runs.
 *
 * Why this exists:
 * - Keeps route page files focused on JSX composition.
 * - Concentrates imperative browser-global behavior in feature-level tooling.
 * - Provides a single, testable contract for external form automation.
 *
 * @param bindings Bridge dependencies bound from the PyTorch training integration hook.
 * @returns Nothing. Registers and cleans up global bridge side effects.
 */
export function usePytorchFormBridge(bindings: PytorchFormBridgeBindings): void {
  const {
    autoDistillEnabled,
    onTrainClick,
    runSweepEnabled,
    setAutoDistillEnabled,
    setBatchSizesInput,
    setDateColumnsInput,
    setDropoutsInput,
    setEpochValuesInput,
    setExcludeColumnsInput,
    setHiddenDimsInput,
    setLearningRatesInput,
    setNumHiddenLayersInput,
    setTargetColumn,
    setTask,
    setTestSizesInput,
    setTrainingMode,
    toggleRunSweep,
  } = bindings;

  useEffect(() => {
    if (typeof window === "undefined") return;

    window.__AIFOLIO_PYTORCH_FORM_BRIDGE__ = {
      applyPatch: (patch) =>
        applyPytorchBridgePatch(patch, {
          setTrainingMode,
          setTargetColumn,
          setTask,
          runSweepEnabled,
          toggleRunSweep,
          setEpochValuesInput,
          setBatchSizesInput,
          setLearningRatesInput,
          setTestSizesInput,
          setHiddenDimsInput,
          setNumHiddenLayersInput,
          setDropoutsInput,
          setExcludeColumnsInput,
          setDateColumnsInput,
          autoDistillEnabled,
          setAutoDistillEnabled,
        }),
      startTrainingRuns: async () => {
        try {
          await flushUiWork();
          await onTrainClick();
          return { status: "ok" };
        } catch (error) {
          const reason = error instanceof Error ? error.message : "Unknown training bridge error.";
          return { status: "error", reason };
        }
      },
    };

    return () => {
      if (window.__AIFOLIO_PYTORCH_FORM_BRIDGE__) {
        delete window.__AIFOLIO_PYTORCH_FORM_BRIDGE__;
      }
    };
  }, [
    autoDistillEnabled,
    onTrainClick,
    runSweepEnabled,
    setAutoDistillEnabled,
    setBatchSizesInput,
    setDateColumnsInput,
    setDropoutsInput,
    setEpochValuesInput,
    setExcludeColumnsInput,
    setHiddenDimsInput,
    setLearningRatesInput,
    setNumHiddenLayersInput,
    setTargetColumn,
    setTask,
    setTestSizesInput,
    setTrainingMode,
    toggleRunSweep,
  ]);
}
