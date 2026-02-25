"use client";

import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { type ReactNode, useMemo, useRef, useState } from "react";

type ResearchRow = {
  id: string;
  dataset: string;
  rows: number;
  updatedAt: string;
  status: "ready" | "processing" | "error";
  owner: string;
  source: string;
  sizeMb: number;
  tags: string;
};

type GenericRow = Record<string, string | number | null>;

/**
 * Build fake rows for UI prototyping.
 * Uses predictable values so you can verify scrolling + virtualization.
 */
const data: ResearchRow[] = Array.from({ length: 1200 }, (_, index) => {
  const id = `ds-${String(index + 1).padStart(4, "0")}`;
  const status: ResearchRow["status"] =
    index % 12 === 0 ? "error" : index % 5 === 0 ? "processing" : "ready";
  return {
    id,
    dataset: `crypto_dataset_${index + 1}.csv`,
    rows: 250 + (index % 5000),
    updatedAt: `2026-01-${String((index % 28) + 1).padStart(2, "0")}`,
    status,
    owner: index % 2 === 0 ? "AIfolio" : "Research",
    source: index % 3 === 0 ? "CoinGecko" : index % 3 === 1 ? "Binance" : "Custom",
    sizeMb: Number((12 + (index % 300) * 0.12).toFixed(1)),
    tags: index % 2 === 0 ? "pricing,30d" : "signals,ml",
  };
});

/**
 * Column definitions for TanStack Table.
 * Includes a checkbox column to prove per-row action wiring.
 */
const researchColumns: ColumnDef<ResearchRow>[] = [
  {
    id: "select",
    header: "",
    cell: ({ row }) => (
      <input
        type="checkbox"
        onChange={(event) => {
          console.log("Row selected:", row.original.id, event.target.checked);
        }}
        aria-label={`Select ${row.original.dataset}`}
      />
    ),
  },
  {
    header: "Dataset",
    accessorKey: "dataset",
  },
  {
    header: "Rows",
    accessorKey: "rows",
  },
  {
    header: "Last Updated",
    accessorKey: "updatedAt",
  },
  {
    header: "Status",
    accessorKey: "status",
    cell: (info) => {
      const value = info.getValue<ResearchRow["status"]>();
      const base =
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium";
      if (value === "ready") {
        return <span className={`${base} bg-emerald-100 text-emerald-700`}>Ready</span>;
      }
      if (value === "processing") {
        return <span className={`${base} bg-amber-100 text-amber-700`}>Processing</span>;
      }
      return <span className={`${base} bg-rose-100 text-rose-700`}>Error</span>;
    },
  },
  {
    header: "Owner",
    accessorKey: "owner",
  },
  {
    header: "Source",
    accessorKey: "source",
  },
  {
    header: "Size (MB)",
    accessorKey: "sizeMb",
  },
  {
    header: "Tags",
    accessorKey: "tags",
  },
];

type DataTableProps = {
  /**
   * Scroll container height (px). Controls visible rows before scrolling.
   */
  height?: number;
  /**
   * Max width (px) for the scroll container. Forces horizontal scroll if columns overflow.
   */
  maxWidth?: number;
  rows?: GenericRow[];
  columns?: string[];
  cellRenderers?: Record<string, (value: GenericRow[keyof GenericRow], row: GenericRow) => ReactNode>;
};

export default function DataTable({
  height = 420,
  maxWidth = 900,
  rows,
  columns,
  cellRenderers = {},
}: DataTableProps) {
  const useGenericRows = Array.isArray(rows);
  const [sorting, setSorting] = useState<SortingState>([]);

  const genericColumns = useMemo<ColumnDef<GenericRow>[]>(() => {
    if (!useGenericRows) return [];
    const keys = (columns ?? []).length > 0 ? columns ?? [] : Object.keys(rows?.[0] ?? {});
    return keys.map((key) => ({
      id: key,
      header: key,
      // Use direct key access so dotted names like "MS.SubClass" are treated
      // as literal keys, not nested access paths.
      accessorFn: (row) => row[key] ?? null,
      cell: ({ row, getValue }) => {
        const value = getValue<GenericRow[keyof GenericRow]>();
        const renderer = cellRenderers[key];
        if (renderer) {
          return renderer(value, row.original);
        }
        return value as ReactNode;
      },
    }));
  }, [cellRenderers, columns, rows, useGenericRows]);

  /**
   * Build the TanStack table instance with the core row model only.
   */
  const table = useReactTable({
    data: useGenericRows ? rows ?? [] : data,
    columns: useGenericRows ? genericColumns : researchColumns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: { sorting },
  });

  /**
   * Scroll container ref used by the virtualizer to read scroll position.
   */
  const parentRef = useRef<HTMLDivElement | null>(null);
  const tableRows = table.getRowModel().rows;

  /**
   * Virtualizer renders only the rows in view to keep large tables fast.
   */
  const rowVirtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 48,
    overscan: 8,
  });

  /**
   * Virtual row metadata + padding to preserve full scroll height.
   */
  const virtualItems = rowVirtualizer.getVirtualItems();
  const totalSize = rowVirtualizer.getTotalSize();

  const paddingTop = virtualItems.length ? virtualItems[0].start : 0;
  const paddingBottom = virtualItems.length
    ? totalSize - virtualItems[virtualItems.length - 1].end
    : 0;

  const headerGroups = table.getHeaderGroups();

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
      <div
        ref={parentRef}
        className="overflow-auto"
        style={{ height, maxWidth }}
      >
        <table className="min-w-[900px] w-full text-left text-sm">
          <thead className="sticky top-0 z-10 bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
            {headerGroups.map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-4 py-3 font-semibold"
                  >
                    {header.isPlaceholder ? null : (
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className="inline-flex items-center gap-1 text-left"
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {header.column.getIsSorted() === "asc" ? "▲" : ""}
                        {header.column.getIsSorted() === "desc" ? "▼" : ""}
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-zinc-200">
            {paddingTop > 0 ? (
              <tr>
                <td style={{ height: paddingTop }} />
              </tr>
            ) : null}
            {virtualItems.map((virtualRow) => {
              const row = tableRows[virtualRow.index];
              return (
                <tr
                  key={row.id}
                  className="hover:bg-zinc-50"
                  style={{ height: virtualRow.size }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-4 py-3 text-zinc-700">
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
            {paddingBottom > 0 ? (
              <tr>
                <td style={{ height: paddingBottom }} />
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
