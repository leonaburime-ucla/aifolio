import type { ChatModelOption } from "@/features/ai-chat/__types__/typescript/chat.types";

export type ModelSelectionResult = {
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
};

export type FallbackSelectionInput = {
  selectedModelId: string | null;
};

export type FallbackSelectionOptions = {
  fallbackModels?: ChatModelOption[];
};

export type FetchedModelsResult = {
  currentModel: string | null;
  models: ChatModelOption[];
};

export type FetchedSelectionInput = {
  selectedModelId: string | null;
  result: FetchedModelsResult;
};
