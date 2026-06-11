import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { usePathnameMock, setSelectedModelIdMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(),
  setSelectedModelIdMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: usePathnameMock,
}));

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    onClick,
    ...props
  }: {
    href: string;
    children: ReactNode;
    onClick?: React.MouseEventHandler<HTMLAnchorElement>;
  }) => (
    <a
      href={href}
      {...props}
      onClick={(event) => {
        event.preventDefault();
        onClick?.(event);
      }}
    >
      {children}
    </a>
  ),
}));

vi.mock("@/features/ag-ui-chat/react/state/zustand/agUiModelStore", () => ({
  useAgUiModelStore: (
    selector: (state: { setSelectedModelId: typeof setSelectedModelIdMock }) => unknown
  ) => selector({ setSelectedModelId: setSelectedModelIdMock }),
}));

import Navbar from "@/ui/patterns/Nav/Navbar";

afterEach(() => {
  cleanup();
});

describe("Navbar", () => {
  beforeEach(() => {
    usePathnameMock.mockReset();
    setSelectedModelIdMock.mockReset();
  });

  it("highlights the active route", () => {
    usePathnameMock.mockReturnValue("/agentic-research");

    render(<Navbar />);

    expect(screen.getByRole("link", { name: "AIFolio" })).toHaveAttribute(
      "href",
      "/"
    );
    expect(screen.getByRole("link", { name: "Agentic Research" }).className).toContain(
      "bg-zinc-300"
    );
    expect(screen.getByRole("link", { name: "AI Chat" }).className).not.toContain(
      "bg-zinc-300"
    );
  });

  it("sets the AG-UI preferred model when the Agentic UI link is clicked", () => {
    usePathnameMock.mockReturnValue("/");

    render(<Navbar />);

    fireEvent.click(screen.getByRole("link", { name: "Agentic UI" }));
    expect(setSelectedModelIdMock).toHaveBeenCalledTimes(1);
  });
});
