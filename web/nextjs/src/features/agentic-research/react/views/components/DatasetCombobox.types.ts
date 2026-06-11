import type { DatasetOption } from "@aifolio/contracts/entities/agentic-research";

export type DatasetComboboxProps = {
  options: DatasetOption[];
  selectedId: string | null;
  onChange: (id: string | null) => void;
};
