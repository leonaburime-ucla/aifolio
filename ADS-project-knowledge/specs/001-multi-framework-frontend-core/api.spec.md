# API Contract Spec: Multi-Framework Frontend Core Extraction

SPEC PACKAGE FILE: `api.spec.md`

- Spec ID: `SPEC-001`
- Feature: `FEAT-001-multi-framework-frontend-core`
- Version: `1.0.0`
- Content Hash: `sha256:placeholder`
- Last Edited: `2026-06-09T17:00:00Z`

## Purpose

This file defines the public API surface of the extracted shared packages -- the TypeScript module exports that consuming framework apps depend on. These are not HTTP endpoints; they are package-level programmatic APIs (exported types, functions, constants, and schemas).

## 1) Package Export Registry

| Export ID | Package | Export Path | Purpose | Stability |
|---|---|---|---|---|
| `CHART_SPEC_TYPE` | contracts | `entities/chart` | ChartSpec type definition | STABLE |
| `CHART_SPEC_SCHEMA` | contracts | `entities/chart` | ChartSpec Zod runtime schema | STABLE |
| `CHART_ACTIONS_PORT` | contracts | `entities/chart` | ChartActionsPort type | STABLE |
| `CHAT_MESSAGE_TYPE` | contracts | `entities/chat` | ChatMessage type definition | STABLE |
| `CHAT_MESSAGE_SCHEMA` | contracts | `entities/chat` | ChatMessage Zod runtime schema | STABLE |
| `CHAT_MODEL_OPTION_TYPE` | contracts | `entities/chat` | ChatModelOption type | STABLE |
| `CHAT_ASSISTANT_PAYLOAD_TYPE` | contracts | `entities/chat` | ChatAssistantPayload type | STABLE |
| `CHAT_HISTORY_MESSAGE_TYPE` | contracts | `entities/chat` | ChatHistoryMessage type | STABLE |
| `CHAT_STATE_TYPE` | contracts | `entities/chat` | ChatState type (observable state shape) | STABLE |
| `CHAT_STATE_ACTIONS_TYPE` | contracts | `entities/chat` | ChatStateActions type (mutation interface) | STABLE |
| `ML_DATASET_OPTION_TYPE` | contracts | `entities/ml` | MlDatasetOption type | STABLE |
| `ML_DATASET_CACHE_ENTRY_TYPE` | contracts | `entities/ml` | MlDatasetCacheEntry type | STABLE |
| `ML_DATASET_STATE_TYPE` | contracts | `entities/ml` | MlDatasetState type (observable state shape) | STABLE |
| `AGUI_TOOL_ACTION_TYPES` | contracts | `entities/ag-ui` | AG-UI tool and action payload types | STABLE |
| `MODEL_SELECTION_RESULT_TYPE` | contracts | `entities/chat` | ModelSelectionResult type | STABLE |
| `RESOLVE_FALLBACK_MODEL` | frontend-core | `features/model-selection` | resolveFallbackModelSelection function | STABLE |
| `RESOLVE_FETCHED_MODEL` | frontend-core | `features/model-selection` | resolveFetchedModelSelection function | STABLE |
| `FALLBACK_CHAT_MODELS` | frontend-core | `features/model-selection` | Default fallback model list constant | STABLE |
| `NORMALIZE_SUBMISSION` | frontend-core | `features/chat-submission` | normalizeSubmissionValue function | STABLE |
| `BUILD_HISTORY_WINDOW` | frontend-core | `features/chat-submission` | buildChatHistoryWindow function | STABLE |
| `CREATE_USER_MESSAGE` | frontend-core | `features/chat-submission` | createUserChatMessage function | STABLE |
| `CREATE_ASSISTANT_MESSAGE` | frontend-core | `features/chat-submission` | createAssistantChatMessage function | STABLE |
| `VALIDATE_TRAINING_INPUT` | frontend-core | `features/ml-validation` | ML training input validation function | STABLE |
| `FORMAT_CHART_DATA` | frontend-core | `features/chart-formatting` | Chart data formatting/normalization | STABLE |
| `ECHART_OPTIONS_BUILDER` | frontend-core | `features/chart-formatting` | ECharts options construction logic | STABLE |

## 2) Package Boundary Rules

| Rule ID | Rule | Enforcement |
|---|---|---|
| `PBR-01` | The `contracts` package must have zero runtime dependencies beyond a validation library (Zod or equivalent) | package.json `dependencies` field audit |
| `PBR-02` | The `frontend-core` package may depend on `contracts` but not on any framework-specific package | package.json `dependencies` + ESLint rule |
| `PBR-03` | Framework apps may depend on both `contracts` and `frontend-core` | workspace dependency graph |
| `PBR-04` | No circular dependency may exist between `contracts` and `frontend-core` | build-order validation |
| `PBR-05` | All exports must be named exports -- no default exports in shared packages | ESLint rule |

## 3) Type Contract Definitions

### 3.1 ChartSpec (from `contracts/entities/chart`)

```typescript
type ChartSpec = {
  id: string;
  title: string;
  description?: string;
  type: "line" | "area" | "bar" | "scatter" | "histogram" | "density"
       | "roc" | "pr" | "errorbar" | "heatmap" | "box" | "violin"
       | "biplot" | "dendrogram" | "surface";
  xKey: string;
  yKeys: string[];
  xLabel?: string;
  yLabel?: string;
  zKey?: string;
  colorKey?: string;
  errorKeys?: Record<string, string>;
  data: Array<Record<string, number | string>>;
  unit?: string;
  currency?: string;
  timeframe?: { start: string; end: string };
  source?: { provider: string; url?: string };
  meta?: { datasetLabel?: string; queryTimeMs?: number };
};
```

### 3.2 ChatMessage (from `contracts/entities/chat`)

```typescript
type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  chartSpec?: ChartSpec | null;
};
```

### 3.3 ChatModelOption (from `contracts/entities/chat`)

```typescript
type ChatModelOption = {
  id: string;
  label: string;
};
```

### 3.4 ChatAssistantPayload (from `contracts/entities/chat`)

```typescript
type ChatAssistantPayload = {
  message: string;
  chartSpec: ChartSpec | ChartSpec[] | null;
};
```

### 3.5 MlDatasetOption (from `contracts/entities/ml`)

```typescript
type MlDatasetOption = {
  id: string;
  label: string;
  description?: string;
};
```

### 3.6 MlDatasetCacheEntry (from `contracts/entities/ml`)

```typescript
type MlDatasetCacheEntry = {
  columns: string[];
  rows: Array<Record<string, string | number | null>>;
  rowCount: number;
  totalRowCount: number;
};
```

### 3.7 ModelSelectionResult (from `contracts/entities/chat`)

```typescript
type ModelSelectionResult = {
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
};
```

### 3.8 ChatState (from `contracts/entities/chat`)

```typescript
type ChatState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  screenFeedback: ScreenFeedback | null;
};
```

### 3.9 MlDatasetState (from `contracts/entities/ml`)

```typescript
type MlDatasetState = {
  datasetOptions: MlDatasetOption[];
  selectedDatasetId: string | null;
  datasetCache: Record<string, MlDatasetCacheEntry>;
  manifestLoaded: boolean;
  isLoadingManifest: boolean;
  isLoadingDataset: boolean;
  error: string | null;
};
```

## 4) Function Contracts

### 4.1 resolveFallbackModelSelection

```typescript
function resolveFallbackModelSelection(
  input: { selectedModelId: string | null },
  options?: { fallbackModels?: ChatModelOption[] }
): ModelSelectionResult;
```

**Behavior:** Returns fallback model list and resolves selected model. If `input.selectedModelId` is non-null, preserves it. Otherwise selects first model from the fallback list.

### 4.2 resolveFetchedModelSelection

```typescript
function resolveFetchedModelSelection(
  input: {
    selectedModelId: string | null;
    result: { models: ChatModelOption[]; currentModel: string | null };
  }
): ModelSelectionResult;
```

**Behavior:** Returns fetched model list and resolves selected model. Priority: `input.selectedModelId` > `input.result.currentModel` > first model in list > null.

### 4.3 normalizeSubmissionValue

```typescript
function normalizeSubmissionValue(
  input: { value: string }
): string | null;
```

**Behavior:** Trims whitespace. Returns null if result is empty string. Returns trimmed value otherwise.

### 4.4 createUserChatMessage / createAssistantChatMessage

```typescript
function createUserChatMessage(
  input: { content: string; id?: string }
): ChatMessage;

function createAssistantChatMessage(
  input: { content: string; id?: string; chartSpec?: ChartSpec | null }
): ChatMessage;
```

**Behavior:** Creates a ChatMessage with role set to "user" or "assistant" respectively, createdAt set to current timestamp, and id generated if not provided.

## 5) Versioning Contract

| Rule | Description |
|---|---|
| Semver for packages | Major: breaking type changes. Minor: new exports. Patch: bug fixes in logic. |
| Type compatibility | Removing a required field or narrowing a union is a MAJOR change. Adding optional fields is MINOR. |
| Function signature | Adding optional parameters is MINOR. Changing return type is MAJOR. |
| Deprecation | Deprecated exports must remain for one major version with `@deprecated` JSDoc. |

## 6) Contract Acceptance Checklist

- [ ] Every export in Section 1 has a corresponding type definition or function contract in Sections 3-4.
- [ ] Every type contract includes all required and optional fields with explicit types.
- [ ] Every function contract specifies input types, output types, and deterministic behavior.
- [ ] Package boundary rules (Section 2) are enforceable via tooling.
- [ ] Versioning contract (Section 5) covers all change categories.
- [ ] No export uses `any`, `unknown`, or untyped escape hatches.
