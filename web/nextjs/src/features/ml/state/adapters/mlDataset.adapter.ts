import {
  useMlDatasetActions,
  useMlDatasetState,
} from "@/features/ml/state/zustand/mlDataStore";

/**
 * Adapter that exposes ML dataset state/actions through an injectable port.
 */
export function useMlDatasetStateAdapter() {
  const state = useMlDatasetState();
  const actions = useMlDatasetActions();

  return {
    state,
    actions,
  };
}
