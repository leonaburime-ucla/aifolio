import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
} from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/ui/components/General/button", () => ({
  Button: (props: ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props} />,
}));

vi.mock("@/ui/components/General/popover", () => ({
  Popover: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/ui/components/General/command", () => ({
  Command: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CommandInput: (props: InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
  CommandEmpty: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CommandGroup: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CommandList: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CommandItem: ({
    children,
    onSelect,
    value,
  }: {
    children: ReactNode;
    onSelect?: (value: string) => void;
    value?: string;
  }) => (
    <button type="button" onClick={() => onSelect?.(value ?? "")}>
      {children}
    </button>
  ),
}));

import CsvDatasetCombobox from "@/ui/patterns/CsvDatasetCombobox";

afterEach(() => {
  cleanup();
});

describe("CsvDatasetCombobox", () => {
  it("shows the placeholder and calls onChange with the selected dataset id", () => {
    const onChange = vi.fn();

    render(
      <CsvDatasetCombobox
        options={[
          { id: "fraud_detection", label: "Fraud Detection", description: "Classification dataset" },
          { id: "house_prices", label: "House Prices" },
        ]}
        selectedId={null}
        onChange={onChange}
      />
    );

    expect(screen.getByRole("combobox")).toHaveTextContent("Select dataset...");
    fireEvent.click(screen.getByRole("button", { name: /fraud detection/i }));
    expect(onChange).toHaveBeenCalledWith("fraud_detection");
  });

  it("renders the selected label when the selection changes", () => {
    render(
      <CsvDatasetCombobox
        options={[
          { id: "fraud_detection", label: "Fraud Detection" },
          { id: "house_prices", label: "House Prices" },
        ]}
        selectedId="house_prices"
        onChange={vi.fn()}
      />
    );

    expect(screen.getByRole("combobox")).toHaveTextContent("House Prices");
  });
});
