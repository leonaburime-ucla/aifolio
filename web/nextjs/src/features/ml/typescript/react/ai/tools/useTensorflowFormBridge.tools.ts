import { useEffect } from "react";
import type {
  TensorflowFormBridge,
  TensorflowFormBridgeBindings,
} from "@/features/ml/__types__/typescript/react/ai/tools/tensorflowFormBridge.tools.types";
import { applyTensorflowBridgePatch } from "@/features/ml/typescript/react/ai/tools/tensorflowFormBridgePatch.tools";

declare global {
  interface Window {
    __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: TensorflowFormBridge;
  }
}

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
 * Registers a controlled global TensorFlow bridge that AI tool calls can use.
 */
export function useTensorflowFormBridge(bindings: TensorflowFormBridgeBindings): void {
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

    window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = {
      applyPatch: (patch) =>
        applyTensorflowBridgePatch(patch, {
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
          return { status: "ok" as const };
        } catch (error) {
          const reason = error instanceof Error ? error.message : "Unknown training bridge error.";
          return { status: "error" as const, reason };
        }
      },
    };

    return () => {
      if (window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__) {
        delete window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
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
