"use client";

import { useMemo, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { DatasetOption } from "@/features/agentic-research/types/agenticResearch.types";

type DatasetComboboxProps = {
  options: DatasetOption[];
  selectedId: string | null;
  onChange: (id: string | null) => void;
};

export default function DatasetCombobox({
  options,
  selectedId,
  onChange,
}: DatasetComboboxProps) {
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
            {selected ? selected.label : "Select dataset..."}
          </span>
          <ChevronsUpDown className="h-4 w-4 text-zinc-400" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[420px] p-0">
        <Command>
          <CommandInput placeholder="Search datasets..." />
          <CommandList>
            <CommandEmpty>No dataset found.</CommandEmpty>
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
