import { createChatStore } from "./chatStore";
import { createChartStore } from "./chartStore";
import { createUIStore } from "./uiStore";

export const StoreManager = {
  ui: createUIStore(),
  chart: createChartStore(),
  chat: createChatStore(),
};

export type StoreManagerType = typeof StoreManager;
