"use client";

import TensorflowTrainingScreen from "@/features/ml/react/views/screens/TensorflowTrainingScreen";

export default function TensorflowAgUiSurface() {
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
            <li>Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.</li>
            <li>Change from customer churn to house prices. Set task to regression, set epochs to 20 and 40, and start training runs.</li>
            <li>Randomize TensorFlow form fields with one value each, and keep the current algorithm.</li>
            <li>Switch the algorithm to entity embeddings, and turn auto-distill on.</li>
          </ol>
        </div>
      </details>
      <TensorflowTrainingScreen />
    </>
  );
}
