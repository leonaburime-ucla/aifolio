import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { Popover, PopoverContent, PopoverTrigger } from "@/ui/components/General/popover";

function PopoverFixture() {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button type="button">Toggle</button>
      </PopoverTrigger>
      <PopoverContent>Popover body</PopoverContent>
    </Popover>
  );
}

describe("Popover", () => {
  it("renders content in a portal when opened", async () => {
    render(<PopoverFixture />);

    expect(screen.queryByText("Popover body")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Toggle" }));
    expect(await screen.findByText("Popover body")).toBeInTheDocument();
  });
});
