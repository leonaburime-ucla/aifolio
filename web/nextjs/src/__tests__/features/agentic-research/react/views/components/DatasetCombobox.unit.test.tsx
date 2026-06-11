import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import DatasetCombobox from "@/features/agentic-research/react/views/components/DatasetCombobox";

const csvComboboxMock = vi.fn(
  ({
    options,
    selectedId,
    onChange,
  }: {
    options: { id: string; label: string; description?: string }[];
    selectedId: string | null;
    onChange: (value: string | null) => void;
  }) => (
    <div data-testid="csv-combobox">
      {selectedId}:{options.map((item) => item.id).join(",")}
    </div>
  )
);

vi.mock("@/ui/patterns/CsvDatasetCombobox", () => ({
  default: (props: {
    options: { id: string; label: string; description?: string }[];
    selectedId: string | null;
    onChange: (value: string | null) => void;
  }) => csvComboboxMock(props),
}));

describe("DatasetCombobox", () => {
  it("forwards options/selectedId/onChange to CsvDatasetCombobox", () => {
    const onChange = vi.fn();
    render(
      <DatasetCombobox
        options={[{ id: "iris", label: "Iris" }]}
        selectedId="iris"
        onChange={onChange}
      />
    );

    expect(screen.getByTestId("csv-combobox").textContent).toBe("iris:iris");
    const props = csvComboboxMock.mock.calls[0]?.[0];
    expect(props.onChange).toBe(onChange);
  });
});
