<template>
  <div class="flex h-[calc(100dvh-64px)] flex-row overflow-hidden bg-zinc-50 text-zinc-900">
    <main class="min-w-0 flex-1 overflow-y-auto py-2">
      <div class="mx-auto flex max-w-5xl flex-col gap-3 px-6">

        <details class="rounded-2xl border border-zinc-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
          <summary class="cursor-pointer text-sm font-semibold text-zinc-900">
            What is AG-UI?
          </summary>
          <div class="mt-3 space-y-3 text-sm text-zinc-700">
            <p>
              AG-UI is a protocol for agent-to-UI actions. Instead of only returning text, an LLM can call
              structured tools that mutate the interface: switch tabs, select datasets, clear/add charts, and
              trigger page workflows.
            </p>
            <p>
              CopilotKit is the runtime bridge that registers those frontend tools and executes them safely in
              this app. It maps model tool calls to typed handlers so chat can control real UI state.
            </p>
            <p>
              In this workspace, chat can orchestrate multi-step flows across tabs by combining navigation and
              feature-specific tools in sequence.
            </p>
            <p>
              References:
              <a href="https://github.com/ag-ui-protocol/ag-ui" target="_blank" rel="noreferrer"
                class="underline decoration-zinc-400 underline-offset-2 hover:text-zinc-900">AG-UI</a>
              |
              <a href="https://github.com/CopilotKit/CopilotKit" target="_blank" rel="noreferrer"
                class="underline decoration-zinc-400 underline-offset-2 hover:text-zinc-900">CopilotKit</a>
            </p>
          </div>
        </details>

        <p class="text-sm font-semibold text-red-600">For best results use Gemini 3.1 Pro Preview</p>

        <div class="sticky top-0 z-20 rounded-2xl border border-zinc-200 bg-white/90 p-2 shadow-sm backdrop-blur-sm">
          <div class="grid grid-cols-2 gap-2 md:grid-cols-4">
            <button
              v-for="tab in tabs"
              :key="tab.id"
              type="button"
              :class="[
                'rounded-xl px-3 py-2 text-sm font-medium transition',
                activeTab === tab.id
                  ? 'bg-zinc-900 text-white'
                  : 'bg-zinc-100 text-zinc-700 hover:bg-zinc-200'
              ]"
              @click="activeTab = tab.id"
            >
              {{ tab.label }}
            </button>
          </div>
        </div>

        <div class="flex items-center justify-between">
          <button
            type="button"
            class="rounded-md border border-emerald-600 bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-emerald-700"
            @click="showTools = true"
          >
            Show Tools
          </button>
        </div>

        <!-- Tools Modal -->
        <Teleport to="body">
          <div
            v-if="showTools"
            class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40 p-4"
            @click.self="showTools = false"
          >
            <div class="w-full max-w-2xl rounded-xl border border-zinc-200 bg-white shadow-2xl">
              <div class="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
                <h3 class="text-sm font-semibold text-zinc-900">
                  Available Tools — {{ activeTabLabel }}
                </h3>
                <button
                  type="button"
                  class="rounded-md border border-zinc-300 px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-50"
                  @click="showTools = false"
                >
                  ✕
                </button>
              </div>
              <div class="max-h-[60vh] overflow-y-auto px-4 py-3">
                <p class="text-xs text-zinc-600">
                  Tools are callable actions the model can invoke for this page to perform structured UI operations.
                </p>
                <ul class="mt-3 space-y-2 text-sm">
                  <li
                    v-for="tool in toolsForTab"
                    :key="tool.name"
                    class="rounded-md border border-zinc-200 px-3 py-2"
                  >
                    <p class="font-mono text-xs text-zinc-800">{{ tool.name }}</p>
                    <p class="text-xs text-zinc-600">{{ tool.description }}</p>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </Teleport>

        <!-- Tab surfaces -->
        <template v-if="activeTab === 'charts'">
          <ChartsWorkspace />
        </template>

        <template v-else-if="activeTab === 'agentic-research'">
          <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary class="cursor-pointer text-[12px] font-semibold">Show Sample Prompts</summary>
            <div class="mt-3 text-[12px]">
              <p class="font-bold text-red-600">Results take 1-2min</p>
              <ol class="mt-3 flex list-decimal flex-col gap-1 pl-5">
                <li>Run PCA Transform</li>
                <li>Run NMF Decomposition and PLSR</li>
                <li>Change the dataset to fraud detection and run Random Forest</li>
              </ol>
            </div>
          </details>
          <AgenticResearchWorkspace @dataset-change="onDatasetChange" />
        </template>

        <template v-else-if="activeTab === 'pytorch'">
          <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary class="cursor-pointer text-[12px] font-semibold">Show Sample Prompts</summary>
            <div class="mt-3 text-[12px]">
              <ol class="flex list-decimal flex-col gap-1 pl-5">
                <li>Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Set batch sizes to 33 and 40, hidden dims to 64 and 96, and dropouts to 0.1 and 0.2.</li>
                <li>Change from customer churn to fraud detection. Set task to classification, choose a different target column, set test sizes to 0.2 and 0.3, and start training runs.</li>
                <li>Randomize PyTorch form fields with one value each, keep the current algorithm, and start training runs.</li>
                <li>Switch the algorithm to calibrated classifier and set sweep values on.</li>
              </ol>
            </div>
          </details>
          <PytorchTrainingScreen />
        </template>

        <template v-else-if="activeTab === 'tensorflow'">
          <details class="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary class="cursor-pointer text-[12px] font-semibold">Show Sample Prompts</summary>
            <div class="mt-3 text-[12px]">
              <ol class="flex list-decimal flex-col gap-1 pl-5">
                <li>Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.</li>
                <li>Change from customer churn to house prices. Set task to regression, set epochs to 20 and 40, and start training runs.</li>
                <li>Randomize TensorFlow form fields with one value each, and keep the current algorithm.</li>
                <li>Switch the algorithm to entity embeddings, and turn auto-distill on.</li>
              </ol>
            </div>
          </details>
          <TensorflowTrainingScreen />
        </template>

      </div>
    </main>

    <div class="flex h-full w-[420px] shrink-0 flex-col overflow-hidden">
      <ChatSidebar :mode="chatMode" :dataset-id="activeDatasetId" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import ChartsWorkspace from "~/features/recharts/components/ChartsWorkspace.vue";
import AgenticResearchWorkspace from "~/features/agentic-research/components/AgenticResearchWorkspace.vue";
import PytorchTrainingScreen from "~/features/ml/components/PytorchTrainingScreen.vue";
import TensorflowTrainingScreen from "~/features/ml/components/TensorflowTrainingScreen.vue";
import ChatSidebar from "~/features/ai-chat/components/ChatSidebar.vue";

type WorkspaceTab = "charts" | "agentic-research" | "pytorch" | "tensorflow";

const tabs: { id: WorkspaceTab; label: string }[] = [
  { id: "charts", label: "Charts" },
  { id: "agentic-research", label: "Agentic Research" },
  { id: "pytorch", label: "PyTorch" },
  { id: "tensorflow", label: "Tensorflow" },
];

const activeTab = ref<WorkspaceTab>("charts");
const showTools = ref(false);
const activeDatasetId = ref<string | null>(null);

const activeTabLabel = computed(() => tabs.find((t) => t.id === activeTab.value)?.label ?? "");

const chatMode = computed(() => activeTab.value === "agentic-research" ? "research" as const : "direct" as const);

const toolsForTab = computed(() => {
  const base = [
    { name: "switch_ag_ui_tab", description: "Switch the active workspace tab" },
    { name: "navigate_to_page", description: "Navigate to another page in the app" },
  ];
  switch (activeTab.value) {
    case "charts":
      return [
        ...base,
        { name: "add_chart_spec", description: "Add a chart to the workspace" },
        { name: "clear_charts", description: "Remove all charts from the workspace" },
      ];
    case "agentic-research":
      return [
        ...base,
        { name: "add_chart_spec", description: "Add a chart to the research workspace" },
        { name: "clear_charts", description: "Clear all research charts" },
        { name: "set_active_dataset", description: "Switch the active dataset" },
      ];
    case "pytorch":
      return [
        ...base,
        { name: "set_pytorch_form_fields", description: "Set PyTorch training form fields" },
        { name: "train_pytorch_model", description: "Start a PyTorch training run" },
      ];
    case "tensorflow":
      return [
        ...base,
        { name: "set_tensorflow_form_fields", description: "Set Tensorflow training form fields" },
        { name: "train_tensorflow_model", description: "Start a Tensorflow training run" },
      ];
    default:
      return base;
  }
});

function onDatasetChange(id: string) {
  activeDatasetId.value = id;
}
</script>
