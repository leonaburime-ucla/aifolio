"use client";

import PytorchTrainingScreen from "@/features/ml/react/views/screens/PytorchTrainingScreen";

export default function PytorchAgUiSurface() {
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
          <ol className="flex list-decimal flex-col gap-1 pl-5">
            <li>Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Set batch sizes to 33 and 40, hidden dims to 64 and 96, and dropouts to 0.1 and 0.2.</li>
            <li>Change from customer churn to fraud detection. Set task to classification, choose a different target column, set test sizes to 0.2 and 0.3, and start training runs.</li>
            <li>Randomize PyTorch form fields with one value each, keep the current algorithm, and start training runs.</li>
            <li>Switch the algorithm to calibrated classifier and set sweep values on.</li>
          </ol>
        </div>
      </details>
      <PytorchTrainingScreen />
    </>
  );
}
