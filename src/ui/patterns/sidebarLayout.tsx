"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

interface NavItem {
  label: string;
  href: string;
}

interface SidebarLayoutProps {
  navItems: NavItem[];
  title?: string;
  children: React.ReactNode;
}

export function SidebarLayout({ navItems, title = "Agentic Console", children }: SidebarLayoutProps) {
  const [isOpen, setIsOpen] = useState(true);
  const pathname = usePathname();

  const toggle = () => setIsOpen((prev) => !prev);

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {isOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={toggle}
          aria-label="Close navigation overlay"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-80 border-r bg-card p-6 shadow-md transition-transform duration-200 md:shadow-lg ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        aria-label="Primary navigation"
      >
        <div className="mb-6 flex items-center justify-between">
          <span className="text-2xl font-semibold">{title}</span>
          <button
            type="button"
            className="rounded-full border px-3 py-2"
            onClick={toggle}
            aria-label="Close navigation"
          >
            ✕
          </button>
        </div>
        <nav className="space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`block rounded-md px-3 py-2 text-sm font-medium transition hover:bg-muted ${
                  isActive ? "bg-muted text-foreground" : "text-muted-foreground"
                }`}
                onClick={() => setIsOpen(false)}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <div className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b bg-card/70 px-6 py-4 shadow-sm">
          <button
            type="button"
            className="flex flex-col gap-1 rounded-md border px-4 py-2"
            onClick={toggle}
            aria-label="Toggle navigation"
          >
            <span className="h-0.5 w-6 bg-foreground" />
            <span className="h-0.5 w-6 bg-foreground" />
            <span className="h-0.5 w-6 bg-foreground" />
          </button>
          {/* <div className="text-lg font-semibold">{title}</div> */}
        </header>

        <main className="flex flex-1 flex-col gap-8 p-6 md:p-10 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
