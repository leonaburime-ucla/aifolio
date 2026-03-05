import type { DatasetOption } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

export type DatasetComboboxProps = {
  options: DatasetOption[];
  selectedId: string | null;
  onChange: (id: string | null) => void;
};
