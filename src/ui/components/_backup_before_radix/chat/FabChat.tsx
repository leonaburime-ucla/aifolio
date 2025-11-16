"use client";

import { useState } from "react";
import { ChatBubbleIcon } from "@radix-ui/react-icons";
import { Button, Card, IconButton } from "@radix-ui/themes";

export function FabChat() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {isOpen && (
        <Card
          className="fixed bottom-24 right-6 z-40 w-80 space-y-3 rounded-xl border border-emerald-500 bg-card p-4 shadow-2xl"
          aria-live="polite"
        >
          <header className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-emerald-600">
                Agent Chat
              </p>
              <h3 className="text-lg font-semibold">Conversation Preview</h3>
            </div>
            <IconButton
              aria-label="Close chat"
              onClick={() => setIsOpen(false)}
              variant="ghost"
            >
              ✕
            </IconButton>
          </header>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>
              Quick placeholder chat body. Wire orchestrators later to stream
              conversation data + diagnostics.
            </p>
            <p className="text-xs">No persistence yet.</p>
          </div>
          <footer className="flex justify-end">
            <Button size="2" variant="surface">
              Start conversation
            </Button>
          </footer>
        </Card>
      )}

      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-500 text-white shadow-lg transition hover:bg-emerald-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300"
        aria-label={isOpen ? "Close chat window" : "Open chat window"}
      >
        <ChatBubbleIcon className="h-6 w-6" />
      </button>
    </>
  );
}
