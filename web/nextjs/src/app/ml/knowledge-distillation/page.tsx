"use client";

import { useState } from "react";
import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import {
  DEFAULT_ML_DATASET_ID,
  ML_WINE_DATASET_OPTIONS,
} from "@/features/ml/api/mlDataApi";

export default function KnowledgeDistillationPage() {
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    DEFAULT_ML_DATASET_ID
  );

  return (
    <div className="min-h-screen bg-white text-zinc-900">
      <div className="mx-auto flex max-w-5xl flex-col gap-4 px-6 py-10">
        <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
          Machine Learning
        </p>
        <h1 className="text-2xl font-semibold text-zinc-900">
          Knowledge Distillation
        </h1>
        <p className="text-sm text-zinc-600">
          Compare teacher vs student error and model-size reduction.
        </p>
        <div className="mt-2 flex max-w-xl flex-col gap-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            CSV Dataset
          </p>
          <CsvDatasetCombobox
            options={ML_WINE_DATASET_OPTIONS}
            selectedId={selectedDatasetId}
            onChange={setSelectedDatasetId}
          />
        </div>
      </div>
    </div>
  );
}
