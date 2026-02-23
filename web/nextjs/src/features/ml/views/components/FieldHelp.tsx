import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/core/views/components/General/popover";

type FieldHelpProps = {
  text: string;
};

/** Inline help popover for form labels in ML training views. */
export function FieldHelp({ text }: FieldHelpProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title={text}
          aria-label="Field help"
          className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-300 text-[10px] font-bold text-zinc-500 hover:bg-zinc-100"
        >
          i
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="text-xs leading-relaxed text-zinc-700">
        {text}
      </PopoverContent>
    </Popover>
  );
}
