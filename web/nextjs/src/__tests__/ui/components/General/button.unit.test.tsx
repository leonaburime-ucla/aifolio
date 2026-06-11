import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Button, buttonVariants } from "@/ui/components/General/button";

describe("Button", () => {
  it("applies the requested variant and size classes", () => {
    render(
      <Button variant="outline" size="lg">
        Save
      </Button>
    );

    const button = screen.getByRole("button", { name: "Save" });
    expect(button).toHaveClass("border");
    expect(button).toHaveClass("h-10");
    expect(button).toHaveClass("px-5");
  });

  it("supports asChild composition without dropping classes", () => {
    render(
      <Button asChild variant="ghost">
        <a href="/docs">Docs</a>
      </Button>
    );

    expect(screen.getByRole("link", { name: "Docs" })).toHaveClass("text-zinc-700");
  });

  it("exposes the variant generator for consumers", () => {
    expect(buttonVariants({ variant: "default", size: "icon" })).toContain("h-9 w-9");
  });
});
