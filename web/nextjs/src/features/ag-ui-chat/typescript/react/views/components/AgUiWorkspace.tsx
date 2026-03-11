"use client";

import AgenticResearchPageScreen from "@/screens/AgenticResearchPage/views/AgenticResearchPageScreen";
import LandingPageScreen from "@/screens/LandingPage/views/LandingPageScreen";
import PyTorchPage from "@/app/ml/pytorch/pytorch";
import TensorFlowPage from "@/app/ml/tensorflow/tensorflow";
import { useAgUiWorkspaceOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/agUiWorkspace.orchestrator";
import AgUiToolsModal from "@/features/ag-ui-chat/typescript/react/views/components/AgUiToolsModal";

/**
 * Unified AG-UI workspace that hosts all major analysis and ML screens.
 */
export default function AgUiWorkspace() {
  const { activeTab, tabs, handleTabClick } = useAgUiWorkspaceOrchestrator();

  return (
    <>
      <div className="sticky top-0 z-20 rounded-2xl border border-zinc-200 bg-white/90 p-2 shadow-sm backdrop-blur-sm">
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => handleTabClick(tab.id)}
              className={`rounded-xl px-3 py-2 text-sm font-medium transition ${activeTab === tab.id
                ? "bg-zinc-900 text-white"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200"
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="my-0.5">
        <AgUiToolsModal activeTab={activeTab} />
      </div>

      {activeTab === "charts" ? (
        <LandingPageScreen showSidebar={false} showTitle={false} horizontalPadding={false} />
      ) : null}
      {activeTab === "agentic-research" ? (
        <>
          <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary className="cursor-pointer text-[12px] font-semibold">
              Show Sample Prompts
            </summary>
            <div className="mt-3 text-[12px]">
              <p className="font-bold text-red-600">Results take 1-2min</p>
              <ol className="mt-3 list-decimal pl-5 flex flex-col gap-1">
                <li>Run PCA Transform</li>
                <li>Run NMF Decomposition and PLSR</li>
                <li>Change the dataset to fraud detection and run Lasso Regression</li>
              </ol>
            </div>
          </details>
          <AgenticResearchPageScreen
            showChatSidebar={false}
            algorithmsAccordionInitiallyOpen={false}
            algorithmsAccordionTitle="ML Algorithms"
            showAlgorithmsResultsCallout={false}
            showAlgorithmsSamplePrompts={false}
          />
        </>
      ) : null}
      {activeTab === "pytorch" ? (
        <>
          <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary className="cursor-pointer text-[12px] font-semibold">
              Show Sample Prompts
            </summary>
            <div className="mt-3 text-[12px]">
              <ol className="list-decimal pl-5 flex flex-col gap-1">
                <li>Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Set batch sizes to 33 and 40, hidden dims to 64 and 96, and dropouts to 0.1 and 0.2.</li>
                <li>Change from customer churn to fraud detection. Set task to classification, choose a different target column, set test sizes to 0.2 and 0.3, and start training runs.</li>
                <li>Randomize PyTorch form fields with one value each, keep the current algorithm, and start training runs.</li>
                <li>Switch the algorithm to calibrated classifier and set sweep values on.</li>
              </ol>
            </div>
          </details>
          <PyTorchPage />
        </>
      ) : null}
      {activeTab === "tensorflow" ? (
        <>
          <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary className="cursor-pointer text-[12px] font-semibold">
              Show Sample Prompts
            </summary>
            <div className="mt-3 text-[12px]">
              <ol className="list-decimal pl-5 flex flex-col gap-1">
                <li>Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.</li>
                <li>Change from customer churn to house prices. Set task to regression, choose a different target column, set epochs to 20 and 40, and start training runs.</li>
                <li>Randomize TensorFlow form fields with one value each, and keep the current algorithm.</li>
                <li>Switch the algorithm to entity embeddings, and turn auto-distill on.</li>
              </ol>
            </div>
          </details>
          <TensorFlowPage />
        </>
      ) : null}
    </>
  );
}
