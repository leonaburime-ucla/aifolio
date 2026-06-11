import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/features/ai-chat/logic/*"],
              message:
                "Use @aifolio/frontend-core/features/* instead. Logic was extracted to the shared package.",
            },
            {
              group: ["@/features/ai-chat/__types__/api.types"],
              message:
                "Use @aifolio/contracts/entities/chat/api instead.",
            },
            {
              group: ["@/features/ai-chat/__types__/logic/*"],
              message:
                "Use @aifolio/contracts/entities/chat instead. Logic types live in the contracts package.",
            },
            {
              group: ["@/features/charts/contracts/chart.types"],
              message:
                "Use @aifolio/contracts/entities/chart instead.",
            },
            {
              group: [
                "@/features/ag-ui-chat/logic/copilotToolResultPresentation.logic",
                "@/features/ag-ui-chat/logic/agUiToolsCatalog.logic",
                "@/features/ag-ui-chat/logic/messagePersistence.util",
                "@/features/ag-ui-chat/logic/copilotMessagePersistence.logic",
                "@/features/ag-ui-chat/logic/agUiWorkspace.logic",
                "@/features/ag-ui-chat/logic/copilotAssistantPayload.util",
                "@/features/ag-ui-chat/config/frontendTools.config",
                "@/features/agentic-research/config/frontendTools.config",
                "@/features/ml/ai/agUi/mlTrainingToolActions.logic",
                "@/features/ml/ai/agUi/mlTrainingFrameworkMetadata.logic",
                "@/features/ml/ai/agUi/mlTrainingTools.logic",
                "@/features/ml/ai/agUi/mlTrainingToolsFlow.logic",
              ],
              message:
                "Use @aifolio/frontend-core/ag-ui instead. Logic was extracted to the shared package.",
            },
            {
              group: [
                "@/features/ml/utils/displayFormat.util",
                "@/features/ml/utils/trainingRuns.util",
                "@/features/ml/utils/bayesianOptimizer.util",
                "@/features/ml/utils/trainingUiShared",
              ],
              message:
                "Use @aifolio/frontend-core/ml-training instead. ML display utils were extracted to the shared package.",
            },
          ],
        },
      ],
    },
  },
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
