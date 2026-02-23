import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import DatasetCombobox from "@/features/agentic-research/views/components/DatasetCombobox";

const captured = {
  props: null as null | Record<string, unknown>,
};

vi.mock("@/core/views/patterns/CsvDatasetCombobox", () => ({
  default: (props: Record<string, unknown>) => {
    captured.props = props;
    return null;
  },
}));

describe("DatasetCombobox", () => {
  it("forwards controlled props to CsvDatasetCombobox", () => {
    const onChange = vi.fn();
    const options = [{ id: "d1", label: "Dataset 1" }];

    render(
      <DatasetCombobox
        options={options}
        selectedId="d1"
        onChange={onChange}
      />
    );

    expect(captured.props?.options).toEqual(options);
    expect(captured.props?.selectedId).toBe("d1");
    expect(captured.props?.onChange).toBe(onChange);
  });
});
