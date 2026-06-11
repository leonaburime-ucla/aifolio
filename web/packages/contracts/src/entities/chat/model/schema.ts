import { z } from "zod";
import { ChartSpecSchema } from "../../chart/index";

export const ChatMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["user", "assistant"]),
  content: z.string(),
  createdAt: z.number(),
  chartSpec: ChartSpecSchema.nullable().optional(),
});

export const ChatModelOptionSchema = z.object({
  id: z.string(),
  label: z.string(),
});

export const ChatAssistantPayloadSchema = z.object({
  message: z.string(),
  chartSpec: z.union([
    ChartSpecSchema,
    z.array(ChartSpecSchema),
    z.null(),
  ]),
});

export const ChatAttachmentSchema = z.object({
  name: z.string(),
  type: z.string(),
  size: z.number(),
  dataUrl: z.string(),
});

export const ChatHistoryMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string(),
  attachments: z.array(ChatAttachmentSchema).optional(),
});

export const ScreenFeedbackSchema = z.object({
  kind: z.enum(["error", "warning", "info"]),
  code: z.string(),
  message: z.string(),
  retryable: z.boolean().optional(),
  actionLabel: z.string().optional(),
});
