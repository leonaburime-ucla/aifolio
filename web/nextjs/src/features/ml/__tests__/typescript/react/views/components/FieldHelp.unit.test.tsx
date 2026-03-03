import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

vi.mock("@/core/views/components/General/popover", () => ({
  Popover: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PopoverContent: ({ children }: { children: ReactNode }) => (
    <div data-testid="popover-content">{children}</div>
  ),
}));

import { FieldHelp } from "@/features/ml/typescript/react/views/components/FieldHelp";

describe("FieldHelp", () => {
  it("renders trigger and help text", () => {
    render(<FieldHelp text="Helpful tip" />);
    expect(screen.getByRole("button", { name: "Field help" })).toBeInTheDocument();
    expect(screen.getByTestId("popover-content")).toHaveTextContent("Helpful tip");
  });
});
