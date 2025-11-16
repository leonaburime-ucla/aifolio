"use client";

import { useState } from "react";
import { Button, Card, IconButton } from "@radix-ui/themes";
import { EditorContent, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";

const MessageGlyph = () => (
  <svg
    viewBox="0 0 24 24"
    role="img"
    aria-hidden="true"
    className="h-7 w-7 stroke-current text-white"
    fill="none"
    strokeWidth="2"
  >
    <path d="M4 6.5c0-1.657 1.567-3 3.5-3h9c1.933 0 3.5 1.343 3.5 3V15c0 1.657-1.567 3-3.5 3H10l-3.5 3v-3H7.5C5.567 18 4 16.657 4 15z" />
  </svg>
);

export function FabChat() {
  const [isOpen, setIsOpen] = useState(false);
  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: "Type your query or drag an image onto the prompt",
      }),
    ],
    editorProps: {
      attributes: {
        class:
          "min-h-[8rem] w-full rounded-xl border border-emerald-100 bg-[rgb(250,248,242)] px-3 py-2 text-sm leading-relaxed focus:outline-none",
      },
      handleDrop: (view, event) => {
        const file = event.dataTransfer?.files?.[0];
        if (file) {
          const droppedPath =
            event.dataTransfer?.getData("text/uri-list") ||
            ("path" in file ? (file as File & { path?: string }).path : "") ||
            file.webkitRelativePath ||
            file.name;
          const text = `'${droppedPath}'`;
          const { tr, selection } = view.state;
          view.dispatch(tr.insertText(text, selection.from, selection.to));
          return true;
        }
        return false;
      },
      handlePaste: (view, event) => {
        const uri =
          event.clipboardData?.getData("text/uri-list") ||
          event.clipboardData?.getData("text/plain");
        if (uri && uri.length > 0) {
          const { tr, selection } = view.state;
          view.dispatch(tr.insertText(`'${uri}'`, selection.from, selection.to));
          return true;
        }
        return false;
      },
    },
    enableInputRules: false,
    immediatelyRender: false,
  });

  return (
    <>
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-40">
          <Card
            className="flex w-96 flex-col gap-3 rounded-2xl border border-emerald-500 bg-white/95 p-4 shadow-2xl"
            aria-live="polite"
            role="dialog"
            aria-label="Chat window"
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
            <div className="space-y-3 rounded-2xl border border-dashed border-emerald-200 bg-[rgb(250,248,242)] p-3 text-sm text-muted-foreground">
              <p className="text-foreground font-medium">AI Agent</p>
              <p>Type your query or drag an image onto the prompt.</p>
              <EditorContent editor={editor} />
            </div>
            <footer className="flex items-center justify-end gap-2">
              <Button size="2" variant="ghost" onClick={() => setIsOpen(false)}>
                Cancel
              </Button>
              <Button size="2" variant="solid" color="green">
                Send
              </Button>
            </footer>
          </Card>
        </div>
      )}

      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed bottom-6 right-6 z-50 flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500 text-white shadow-lg transition hover:bg-emerald-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300"
        aria-label={isOpen ? "Close chat window" : "Open chat window"}
        aria-expanded={isOpen}
      >
        <MessageGlyph />
      </button>
    </>
  );
}
