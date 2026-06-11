"use client";

import AgenticResearchWorkspaceSurface from "@/features/agentic-research/react/views/screens/AgenticResearchWorkspaceSurface";

export default function AgenticResearchAgUiSurface() {
  return (
    <>
      <details
        className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black"
        open
      >
        <summary className="cursor-pointer text-[12px] font-semibold">
          Show Sample Prompts
        </summary>
        <div className="mt-3 text-[12px]">
          <p className="font-bold text-red-600">Results take 1-2min</p>
          <ol className="mt-3 flex list-decimal flex-col gap-1 pl-5">
            <li>Run PCA Transform</li>
            <li>Run NMF Decomposition and PLSR</li>
            <li>Change the dataset to fraud detection and run Random Forest</li>
          </ol>
        </div>
      </details>
      <AgenticResearchWorkspaceSurface
        algorithmsAccordionInitiallyOpen={false}
        algorithmsAccordionTitle="ML Algorithms"
        showAlgorithmsResultsCallout={false}
        showAlgorithmsSamplePrompts={false}
      />
    </>
  );
}
