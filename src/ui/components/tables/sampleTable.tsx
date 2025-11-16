"use client";

import * as React from "react";
import {
  ColumnDef,
  ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  VisibilityState,
} from "@tanstack/react-table";
import { ArrowUpDown, ChevronDown, MoreHorizontal } from "lucide-react";
import {
  Button,
  Checkbox,
  DropdownMenu,
  Select,
  Table,
  TextField,
} from "@radix-ui/themes";
import { Payment, payments } from "./paymentData";

const columns: ColumnDef<Payment>[] = [
  {
    id: "select",
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() ||
          (table.getIsSomePageRowsSelected() && "indeterminate")
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label="Select all"
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        aria-label="Select row"
      />
    ),
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => (
      <div className="capitalize">{row.getValue("status")}</div>
    ),
  },
  {
    accessorKey: "email",
    header: ({ column }) => (
      <Button
        variant="ghost"
        className="gap-2 px-0 text-muted-foreground hover:text-foreground"
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
      >
        Email
        <ArrowUpDown className="h-4 w-4" />
      </Button>
    ),
    cell: ({ row }) => <div className="lowercase">{row.getValue("email")}</div>,
  },
  {
    accessorKey: "amount",
    header: () => <div className="text-right">Amount</div>,
    cell: ({ row }) => {
      const amount = parseFloat(row.getValue("amount"));
      const formatted = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(amount);

      return <div className="text-right font-medium">{formatted}</div>;
    },
  },
  {
    accessorKey: "customer",
    header: "Customer",
    cell: ({ row }) => <div className="font-medium">{row.getValue("customer")}</div>,
  },
  {
    accessorKey: "method",
    header: "Method",
    cell: ({ row }) => <div className="uppercase text-xs tracking-wide">{row.getValue("method")}</div>,
  },
  {
    accessorKey: "region",
    header: "Region",
    cell: ({ row }) => <div>{row.getValue("region")}</div>,
  },
  {
    accessorKey: "riskScore",
    header: () => <div className="text-right">Risk</div>,
    cell: ({ row }) => {
      const score = parseFloat(row.getValue("riskScore"));
      return <div className="text-right">{score.toFixed(2)}</div>;
    },
  },
  {
    id: "actions",
    enableHiding: false,
    cell: ({ row }) => {
      const payment = row.original;

      return (
        <DropdownMenu.Root>
          <DropdownMenu.Trigger>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="end">
            <DropdownMenu.Label>Actions</DropdownMenu.Label>
            <DropdownMenu.Item
              onClick={() => navigator.clipboard.writeText(payment.id)}
            >
              Copy payment ID
            </DropdownMenu.Item>
            <DropdownMenu.Separator />
            <DropdownMenu.Item>View customer</DropdownMenu.Item>
            <DropdownMenu.Item>View payment details</DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      );
    },
  },
];

const DEFAULT_COLUMN_WIDTH = "min-w-[8rem]";

const getColumnWidthClass = (columnId?: string) => {
  if (columnId === "select") {
    return "min-w-[4rem]";
  }

  if (columnId === "actions") {
    return "min-w-[8rem]";
  }

  return DEFAULT_COLUMN_WIDTH;
};

export function SamplePaymentsTable() {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});

  const table = useReactTable({
    data: payments,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
    },
  });

  return (
    <section className="w-full space-y-4 rounded-xl border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            CSV Dataset
          </p>
          <div className="text-sm text-muted-foreground">
            Select a dataset to wire charts later.
          </div>
        </div>
        <Select.Root>
          <Select.Trigger placeholder="Choose CSV" />
          <Select.Content>
            <Select.Item value="none" disabled>
              No datasets yet
            </Select.Item>
          </Select.Content>
        </Select.Root>
      </div>
      <header className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div>
          <h2 className="text-xl font-semibold">Recent Payments</h2>
          <p className="text-sm text-muted-foreground">
            Demo data table using TanStack + shadcn primitives.
          </p>
        </div>
        <div className="flex w-full items-center gap-2 sm:ml-auto sm:w-auto">
          <TextField.Root
            className="max-w-sm"
            placeholder="Filter emails..."
            value={(table.getColumn("email")?.getFilterValue() as string) ?? ""}
            onChange={(event) =>
              table.getColumn("email")?.setFilterValue(event.target.value)
            }
          />
          <DropdownMenu.Root>
            <DropdownMenu.Trigger>
              <Button variant="outline" className="ml-auto inline-flex items-center gap-2">
                Columns <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              {table
                .getAllColumns()
                .filter((column) => column.getCanHide())
                .map((column) => (
                  <DropdownMenu.CheckboxItem
                    key={column.id}
                    checked={column.getIsVisible()}
                    onCheckedChange={(value) =>
                      column.toggleVisibility(!!value)
                    }
                  >
                    {column.id}
                  </DropdownMenu.CheckboxItem>
                ))}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </header>
      <div className="max-h-96 overflow-x-auto overflow-y-auto rounded-md border">
        <Table.Root>
          <Table.Header>
            {table.getHeaderGroups().map((headerGroup) => (
              <Table.Row key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Table.ColumnHeaderCell
                    key={header.id}
                    className={getColumnWidthClass(header.column.id)}
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                  </Table.ColumnHeaderCell>
                ))}
              </Table.Row>
            ))}
          </Table.Header>
          <Table.Body>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <Table.Row
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                >
                  {row.getVisibleCells().map((cell) => (
                    <Table.Cell
                      key={cell.id}
                      className={getColumnWidthClass(cell.column.id)}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </Table.Cell>
                  ))}
                </Table.Row>
              ))
            ) : (
              <Table.Row>
                <Table.Cell colSpan={columns.length} className="h-24 text-center">
                  No results.
                </Table.Cell>
              </Table.Row>
            )}
          </Table.Body>
        </Table.Root>
      </div>
      <div className="text-muted-foreground text-sm">
        Showing {table.getRowModel().rows.length} row(s).
      </div>
    </section>
  );
}
