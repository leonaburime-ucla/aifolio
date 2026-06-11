import { onMounted } from "vue";
import { useChartStore } from "~/composables/useChartStore";
import { useChatSidebar } from "../model/useChatSidebar";
import type { UseChatSidebarOptions } from "../model/useChatSidebar";

export type UseChatSidebarOrchestratorOptions = Omit<UseChatSidebarOptions, "onChartSpec">;

export function useChatSidebarOrchestrator(options: UseChatSidebarOrchestratorOptions) {
  const chartStore = useChartStore();

  const model = useChatSidebar({
    ...options,
    onChartSpec: (spec: any) => {
      const yKeys: string[] = spec.yKeys ?? (spec.yKey ? [spec.yKey] : []);
      chartStore.addChartSpec({
        id: spec.id || Date.now().toString(36),
        title: spec.title || "Chart",
        type: spec.type || "line",
        xKey: spec.xKey || "x",
        yKeys,
        xLabel: spec.xLabel,
        yLabel: spec.yLabel,
        data: spec.data,
        description: spec.description,
      });
    },
  });

  onMounted(model.loadModels);

  return {
    messages: model.messages,
    inputValue: model.inputValue,
    isSending: model.isSending,
    modelOptions: model.modelOptions,
    selectedModelId: model.selectedModelId,
    screenFeedback: model.screenFeedback,
    messagesEl: model.messagesEl,
    submit: model.submit,
    handleHistory: model.handleHistory,
  };
}
