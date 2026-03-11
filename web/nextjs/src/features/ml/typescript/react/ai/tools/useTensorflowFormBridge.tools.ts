import { useEffect, useRef } from "react";
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
  const bindingsRef = useRef(bindings);

  useEffect(() => {
    // Keep global bridge handlers bound to the latest hook callbacks/state.
    bindingsRef.current = bindings;
  }, [bindings]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = {
      applyPatch: (patch) => {
        const current = bindingsRef.current;
        return applyTensorflowBridgePatch(patch, {
          setDatasetId: current.setDatasetId,
          setTrainingMode: current.setTrainingMode,
          setTargetColumn: current.setTargetColumn,
          setTask: current.setTask,
          runSweepEnabled: current.runSweepEnabled,
          toggleRunSweep: current.toggleRunSweep,
          setEpochValuesInput: current.setEpochValuesInput,
          setBatchSizesInput: current.setBatchSizesInput,
          setLearningRatesInput: current.setLearningRatesInput,
          setTestSizesInput: current.setTestSizesInput,
          setHiddenDimsInput: current.setHiddenDimsInput,
          setNumHiddenLayersInput: current.setNumHiddenLayersInput,
          setDropoutsInput: current.setDropoutsInput,
          setExcludeColumnsInput: current.setExcludeColumnsInput,
          setDateColumnsInput: current.setDateColumnsInput,
          autoDistillEnabled: current.autoDistillEnabled,
          setAutoDistillEnabled: current.setAutoDistillEnabled,
        });
      },
      startTrainingRuns: async () => {
        try {
          await flushUiWork();
          await bindingsRef.current.onTrainClick();
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
  }, []);
}
