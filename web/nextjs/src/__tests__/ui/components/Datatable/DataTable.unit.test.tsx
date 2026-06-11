import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: vi.fn((options: { count: number }) => ({
    getVirtualItems: () =>
      Array.from({ length: options.count }, (_, index) => ({
        index,
        size: 48,
        start: index * 48,
        end: (index + 1) * 48,
      })),
    getTotalSize: () => options.count * 48,
  })),
}));

import DataTable from "@/ui/components/Datatable/DataTable";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("DataTable", () => {
  it("renders generic rows with custom cell renderers", () => {
    render(
      <DataTable
        height={240}
        rows={[
          { dataset: "fraud.csv", rows: 100 },
          { dataset: "housing.csv", rows: 50 },
        ]}
        columns={["dataset", "rows"]}
        cellRenderers={{
          rows: (value) => <span>{`rows:${String(value)}`}</span>,
        }}
      />
    );

    expect(screen.getByText("fraud.csv")).toBeInTheDocument();
    expect(screen.getByText("housing.csv")).toBeInTheDocument();
    expect(screen.getByText("rows:100")).toBeInTheDocument();
    expect(screen.getByText("rows:50")).toBeInTheDocument();
  });

  it("renders the default research table and wires row selection logging", () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    render(<DataTable height={240} />);

    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(screen.getByLabelText("Select crypto_dataset_1.csv")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Select crypto_dataset_1.csv"));
    expect(logSpy).toHaveBeenCalledWith("Row selected:", "ds-0001", true);
  });

  it("sorts generic rows when a header is clicked", () => {
    render(
      <DataTable
        height={240}
        rows={[
          { dataset: "zeta.csv", rows: 10 },
          { dataset: "alpha.csv", rows: 20 },
        ]}
        columns={["dataset", "rows"]}
      />
    );

    fireEvent.click(screen.getAllByRole("button", { name: "dataset" })[0]);

    const bodyRows = screen.getAllByRole("row").slice(1);
    expect(within(bodyRows[0]).getByText("alpha.csv")).toBeInTheDocument();
    expect(within(bodyRows[1]).getByText("zeta.csv")).toBeInTheDocument();
  });
});
