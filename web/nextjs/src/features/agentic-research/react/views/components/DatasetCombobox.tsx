"use client";

import CsvDatasetCombobox from "@/ui/patterns/CsvDatasetCombobox";
import type { DatasetComboboxProps } from "@/features/agentic-research/react/views/components/DatasetCombobox.types";

export default function DatasetCombobox({
  options,
  selectedId,
  onChange,
}: DatasetComboboxProps) {
  return <CsvDatasetCombobox options={options} selectedId={selectedId} onChange={onChange} />;
}
