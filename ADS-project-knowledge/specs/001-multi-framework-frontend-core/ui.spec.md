# UI Contract Spec: Multi-Framework Frontend Core Extraction

SPEC PACKAGE FILE: `ui.spec.md`

- Spec ID: `SPEC-001`
- Feature: `FEAT-001-multi-framework-frontend-core`
- Version: `1.0.0`
- Content Hash: `sha256:placeholder`
- Last Edited: `2026-06-09T17:00:00Z`

## Purpose

Defines the observable UI behavioral contracts that every framework implementation must satisfy. These are NOT component prop definitions (those are framework-specific). These are behavioral parity requirements: given the same state and user interactions, all implementations must produce equivalent observable outcomes.

## 1) Behavioral Parity Registry

| Behavior ID | Feature Area | Description | Parity Level |
|---|---|---|---|
| `BEH-CHAT-01` | AI Chat | Message submission flow | STRICT |
| `BEH-CHAT-02` | AI Chat | Model selection display and interaction | STRICT |
| `BEH-CHAT-03` | AI Chat | Input history navigation | STRICT |
| `BEH-CHAT-04` | AI Chat | Chart rendering from assistant response | STRICT |
| `BEH-ML-01` | ML Training | Dataset selection and loading | STRICT |
| `BEH-ML-02` | ML Training | Training input validation feedback | STRICT |
| `BEH-ML-03` | ML Training | Training run progress display | RELAXED |
| `BEH-CHART-01` | Charts | ChartSpec rendering to visual output | STRICT |
| `BEH-CHART-02` | Charts | Chart type handling for all 15 types | STRICT |
| `BEH-AGUI-01` | AG-UI | Tool action payload construction | STRICT |
| `BEH-AGUI-02` | AG-UI | Workspace state coordination | RELAXED |

**Parity Levels:**
- `STRICT`: Observable output must be functionally identical across frameworks. Same inputs produce same state transitions, same API calls, same rendered data structures.
- `RELAXED`: Business logic outcomes must match. Timing, animation, and intermediate rendering states may differ between frameworks.

## 2) Chat Feature Behavioral Contracts

### BEH-CHAT-01: Message Submission Flow

**Observable behavior sequence:**
1. User enters non-empty text in the chat input.
2. User triggers submit (button click or Enter key).
3. System trims input, validates non-empty.
4. System adds user message to message list with role "user" and current timestamp.
5. System sets sending state to true.
6. System sends message to backend API with conversation history window.
7. On success: system adds assistant message to list, sets sending to false.
8. On failure: system sets sending to false, displays error feedback.

**Parity assertions:**
- Message list after submission must contain the same messages in the same order across all frameworks.
- User message content must be the trimmed input value.
- API request payload must be identical (same history window construction logic via shared `buildChatHistoryWindow`).

### BEH-CHAT-02: Model Selection

**Observable behavior:**
1. On mount, system fetches available models.
2. If fetch succeeds: model options are set from response, selected model resolved via `resolveFetchedModelSelection`.
3. If fetch fails: fallback models applied via `resolveFallbackModelSelection`.
4. User can change selected model.
5. Selected model ID is included in all subsequent API requests.

**Parity assertions:**
- Model resolution logic is identical (shared pure function).
- Selected model persists across message submissions within a session.
- Fallback behavior is identical when API fails.

### BEH-CHAT-03: Input History Navigation

**Observable behavior:**
1. Up arrow in empty/current input: cursor moves to most recent history entry.
2. Up arrow at history position N: cursor moves to N+1 (older).
3. Down arrow at history position N (N > 0): cursor moves to N-1 (newer).
4. Down arrow at position 0: cursor resets to null (current empty input).
5. Any manual text modification: cursor resets to null.
6. Submit: value added to history at position 0, cursor reset.

**Parity assertions:**
- History cursor movement produces identical `selectCurrentHistoryValue` output for the same history array and cursor position.
- Manual modification detection mechanism is framework-specific but reset behavior is identical.

### BEH-CHAT-04: Chart Rendering from Assistant Response

**Observable behavior:**
1. Assistant response payload contains `chartSpec` (single or array).
2. System stores chartSpec(s) via `addChartSpec` action.
3. Chart component receives ChartSpec and renders appropriate visualization.
4. Chart type determines which renderer/configuration is used.

**Parity assertions:**
- ChartSpec data passed to the charting library must be identical across frameworks.
- Chart type routing (which renderer handles which `type` value) must be consistent.
- The 15 chart types must all be handled (even if some display a placeholder for unsupported types).

## 3) ML Training Behavioral Contracts

### BEH-ML-01: Dataset Selection and Loading

**Observable behavior:**
1. On mount, system fetches dataset manifest.
2. System populates dataset options from manifest.
3. User selects a dataset.
4. System fetches dataset content (columns + rows).
5. System caches fetched content.
6. Subsequent selection of cached dataset skips fetch.

**Parity assertions:**
- Dataset options displayed must match the manifest response identically.
- Cache hit/miss behavior must be identical (same cache key logic).
- Loading states must transition in the same sequence.

### BEH-ML-02: Training Input Validation Feedback

**Observable behavior:**
1. User fills training configuration (framework, dataset, hyperparameters).
2. Before submission, validation runs against all inputs.
3. Validation errors are specific: field name + error description.
4. Valid inputs are sent to the backend training API.

**Parity assertions:**
- Validation logic is shared (extracted pure function). Same inputs produce same validation results.
- Error messages and error codes are identical across frameworks.
- A valid/invalid determination is identical for the same inputs.

## 4) Chart Behavioral Contracts

### BEH-CHART-01: ChartSpec to Visual Output

**Observable behavior:**
1. Component receives a ChartSpec object.
2. Component extracts data array, xKey, yKeys, and chart type.
3. Component constructs charting library configuration.
4. Chart renders with correct axes, data series, and labels.

**Parity assertions:**
- Data transformation from ChartSpec to chart library input must use the shared `formatChartData` logic.
- The same ChartSpec must produce the same chart library configuration object (tested via shared contract tests).
- Axis labels, series names, and data point values must be identical.

### BEH-CHART-02: Chart Type Handling

**Required chart types and their render category:**

| Type | Render Category | Notes |
|---|---|---|
| `line` | Cartesian | Standard line chart |
| `area` | Cartesian | Filled area below line |
| `bar` | Cartesian | Vertical bars |
| `scatter` | Cartesian | Point cloud |
| `histogram` | Cartesian | Frequency distribution |
| `density` | Cartesian | Continuous probability |
| `roc` | Cartesian | ROC curve (x: FPR, y: TPR) |
| `pr` | Cartesian | Precision-Recall curve |
| `errorbar` | Cartesian | Points with error bars |
| `heatmap` | Grid | 2D color matrix |
| `box` | Statistical | Box-and-whisker |
| `violin` | Statistical | Distribution shape |
| `biplot` | Cartesian | PCA biplot |
| `dendrogram` | Hierarchical | Tree clustering |
| `surface` | 3D | 3D surface plot |

**Parity assertion:** Each framework app must handle all 15 types. If a charting library does not support a type natively, the app must display a labeled placeholder indicating the unsupported type rather than crashing or rendering nothing.

## 5) AG-UI Behavioral Contracts

### BEH-AGUI-01: Tool Action Payload Construction

**Observable behavior:**
1. AG-UI workspace registers available tool actions.
2. When the AI agent invokes a tool, the system constructs the appropriate payload.
3. Payload construction uses shared logic for normalization and validation.
4. Constructed payload is dispatched to the appropriate handler.

**Parity assertions:**
- Payload shape for each tool action type must conform to the shared contract types.
- Normalization logic (trimming, default injection, type coercion) must produce identical results.
- Invalid payloads must be rejected with the same validation errors.

## 6) Shared Contract Test Requirements

Every behavioral contract above must have at least one test in the shared contract test suite. Tests are structured as:

```typescript
// Pseudocode structure for shared behavioral test
describe("BEH-CHAT-01: Message Submission", () => {
  it("must produce a user message with trimmed content and role 'user'", () => {
    const result = createUserChatMessage({ content: "  hello  " });
    expect(result.role).toBe("user");
    expect(result.content).toBe("hello");
    expect(result.createdAt).toBeTypeOf("number");
  });
});
```

Framework-specific tests verify that the framework's state manager + UI layer produces the correct calls to shared logic. Shared contract tests verify the logic itself.

## 7) Accessibility Parity Requirements

| Requirement | Parity Level | Notes |
|---|---|---|
| Chat input is keyboard-operable (Enter to submit, Up/Down for history) | STRICT | All frameworks must implement same key bindings |
| Chart visualizations have accessible descriptions (aria-label or equivalent) | STRICT | Description derived from ChartSpec.title and ChartSpec.description |
| Loading states are announced to screen readers | RELAXED | Mechanism varies (aria-live, role=status, etc.) but announcement must occur |
| Error messages are associated with their triggering control | STRICT | Validation errors linked to input fields |
| Model selection is operable via keyboard | STRICT | Standard select/combobox keyboard interaction |

## 8) Acceptance Checklist

- [ ] Each behavioral contract has a unique ID and is categorized by parity level.
- [ ] STRICT parity contracts have explicit parity assertions that are testable.
- [ ] RELAXED parity contracts specify which aspects must match and which may differ.
- [ ] All 15 chart types are accounted for in BEH-CHART-02.
- [ ] Accessibility requirements are testable and specify parity level.
- [ ] Shared contract test structure is defined for each behavioral contract.
