import Link from "next/link";

/**
 * Primary site navigation.
 */
export default function Navbar() {
  return (
    <nav className="sticky top-0 z-40 border-b border-zinc-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-6 text-sm text-zinc-900">
          <Link
            href="/"
            className="rounded-md px-2 py-1 transition hover:bg-zinc-100"
          >
            AI Chat
          </Link>

          <Link
            href="/agentic-research"
            className="rounded-md px-2 py-1 transition hover:bg-zinc-100"
          >
            Agentic Research
          </Link>
          <div className="group relative">
            <button
              type="button"
              className="flex items-center gap-1 rounded-md px-2 py-1 transition hover:bg-zinc-100"
            >
              Machine Learning
              <span className="text-xs">▾</span>
            </button>
            <div className="invisible absolute left-0 mt-2 w-48 rounded-lg border border-zinc-200 bg-white p-2 text-sm text-zinc-700 shadow-lg opacity-0 transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
              <Link
                href="/ml/pytorch"
                className="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
              >
                PyTorch
              </Link>
              <Link
                href="/ml/tensorflow"
                className="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
              >
                TensorFlow
              </Link>

            </div>
          </div>
          <Link
            href="/ag-ui"
            className="rounded-md px-2 py-1 transition hover:bg-zinc-100"
          >
            Agentic UI
          </Link>
        </div>

        <Link href="/" className="text-sm font-semibold text-zinc-900">
          AIfolio
        </Link>
      </div>
    </nav>
  );
}
