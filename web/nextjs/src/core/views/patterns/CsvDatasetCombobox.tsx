"use client";

import { useMemo, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/core/views/components/General/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/core/views/components/General/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/core/views/components/General/popover";

export type CsvDatasetOption = {
  id: string;
  label: string;
  description?: string;
};

type CsvDatasetComboboxProps = {
  options: CsvDatasetOption[];
  selectedId: string | null;
  onChange: (id: string | null) => void;
  placeholder?: string;
  emptyMessage?: string;
  searchPlaceholder?: string;
};

export default function CsvDatasetCombobox({
  options,
  selectedId,
  onChange,
  placeholder = "Select dataset...",
  emptyMessage = "No dataset found.",
  searchPlaceholder = "Search datasets...",
}: CsvDatasetComboboxProps) {
  const [open, setOpen] = useState(false);

  const selected = useMemo(
    () => options.find((option) => option.id === selectedId) ?? null,
    [options, selectedId]
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full max-w-md justify-between"
        >
          <span className="truncate text-left text-xs font-medium text-zinc-700">
            {selected ? selected.label : placeholder}
          </span>
          <ChevronsUpDown className="h-4 w-4 text-zinc-400" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[420px] p-0">
        <Command>
          <CommandInput placeholder={searchPlaceholder} />
          <CommandList>
            <CommandEmpty>{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {options.map((option) => (
                <CommandItem
                  key={option.id}
                  value={option.id}
                  onSelect={(value) => {
                    onChange(value || null);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedId === option.id ? "opacity-100" : "opacity-0"
                    )}
                  />
                  <div className="flex flex-col">
                    <span className="text-sm text-zinc-800">{option.label}</span>
                    {option.description ? (
                      <span className="text-xs text-zinc-500">
                        {option.description}
                      </span>
                    ) : null}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
