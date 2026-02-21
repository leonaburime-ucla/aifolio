"use client";

import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import type { DatasetOption } from "@/features/agentic-research/types/agenticResearch.types";

type DatasetComboboxProps = {
  options: DatasetOption[];
  selectedId: string | null;
  onChange: (id: string | null) => void;
};

export default function DatasetCombobox({
  options,
  selectedId,
  onChange,
}: DatasetComboboxProps) {
  return <CsvDatasetCombobox options={options} selectedId={selectedId} onChange={onChange} />;
}
