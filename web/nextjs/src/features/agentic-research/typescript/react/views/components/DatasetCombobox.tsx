"use client";

import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import type { DatasetComboboxProps } from "@/features/agentic-research/__types__/typescript/react/views/datasetCombobox.types";

export default function DatasetCombobox({
  options,
  selectedId,
  onChange,
}: DatasetComboboxProps) {
  return <CsvDatasetCombobox options={options} selectedId={selectedId} onChange={onChange} />;
}
