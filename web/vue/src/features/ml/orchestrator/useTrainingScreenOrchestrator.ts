import { onMounted } from "vue";
import { useTrainingScreen } from "../model/useTrainingScreen";
import type { UseTrainingScreenOptions } from "../model/useTrainingScreen";

export type UseTrainingScreenOrchestratorOptions = UseTrainingScreenOptions;

export function useTrainingScreenOrchestrator(options: UseTrainingScreenOrchestratorOptions) {
  const screen = useTrainingScreen(options);

  onMounted(screen.loadManifest);

  const { loadManifest: _, ...exposed } = screen;
  return exposed;
}
