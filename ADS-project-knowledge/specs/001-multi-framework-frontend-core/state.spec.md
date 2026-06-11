# State Contract Spec: Multi-Framework Frontend Core Extraction

SPEC PACKAGE FILE: `state.spec.md`

- Spec ID: `SPEC-001`
- Feature: `FEAT-001-multi-framework-frontend-core`
- Version: `1.0.0`
- Content Hash: `sha256:placeholder`
- Last Edited: `2026-06-09T17:00:00Z`

## Purpose

Defines the portable state contracts that any framework implementation must satisfy. These contracts specify the observable state shape, legal transitions, and invariants -- not the implementation mechanism. Zustand (Next.js), Pinia (Vue), ngrx (Angular), and Svelte stores are all valid implementations of these contracts.

## 1) Chat State Shape (Portable Contract)

| Field | Type | Nullable | Initial Value | Description |
|---|---|---|---|---|
| `messages` | `array<ChatMessage>` | no | `[]` | Conversation message list |
| `inputHistory` | `array<string>` | no | `[]` | User input history for arrow-key recall |
| `historyCursor` | `integer` | yes | `null` | Current position in input history |
| `isSending` | `boolean` | no | `false` | Whether a message send is in-flight |
| `modelOptions` | `array<ChatModelOption>` | no | `[]` | Available AI model choices |
| `selectedModelId` | `string` | yes | `null` | Currently selected model |
| `isModelsLoading` | `boolean` | no | `false` | Whether model list is being fetched |
| `screenFeedback` | `ScreenFeedback` | yes | `null` | Transient UI feedback state |

## 2) ML Dataset State Shape (Portable Contract)

| Field | Type | Nullable | Initial Value | Description |
|---|---|---|---|---|
| `datasetOptions` | `array<MlDatasetOption>` | no | `[]` | Available dataset choices |
| `selectedDatasetId` | `string` | yes | `null` | Currently selected dataset |
| `datasetCache` | `Record<string, MlDatasetCacheEntry>` | no | `{}` | Cached dataset contents keyed by ID |
| `manifestLoaded` | `boolean` | no | `false` | Whether the dataset manifest has been fetched |
| `isLoadingManifest` | `boolean` | no | `false` | Manifest fetch in-flight |
| `isLoadingDataset` | `boolean` | no | `false` | Dataset content fetch in-flight |
| `error` | `string` | yes | `null` | Last error message |

## 3) Chat State Transition Catalog

| Transition | Trigger | Precondition | State Changes | Postcondition |
|---|---|---|---|---|
| `ADD_MESSAGE` | User sends or assistant responds | none | Append message to `messages` | `messages.length` increased by 1 |
| `ADD_INPUT_TO_HISTORY` | User submits non-empty input | input is non-empty after trim | Prepend to `inputHistory`, reset `historyCursor` to null | `inputHistory[0]` equals submitted value |
| `MOVE_HISTORY_CURSOR_UP` | User presses up arrow in input | `inputHistory.length > 0` | Increment cursor (or set to 0 if null) | `historyCursor` is within `[0, inputHistory.length - 1]` |
| `MOVE_HISTORY_CURSOR_DOWN` | User presses down arrow in input | `historyCursor` is not null | Decrement cursor (or set to null if at 0) | `historyCursor` is null or within `[0, inputHistory.length - 1]` |
| `RESET_HISTORY_CURSOR` | User modifies input manually | none | Set `historyCursor` to null | `historyCursor === null` |
| `SET_SENDING_TRUE` | Message send initiated | `isSending === false` | `isSending = true` | `isSending === true` |
| `SET_SENDING_FALSE` | Message send completes or fails | `isSending === true` | `isSending = false` | `isSending === false` |
| `SET_MODEL_OPTIONS` | Model list fetched | none | Replace `modelOptions` | `modelOptions` matches fetched list |
| `SET_SELECTED_MODEL` | User selects model or fallback applied | none | Update `selectedModelId` | `selectedModelId` is the new value |
| `SET_MODELS_LOADING` | Model fetch starts/ends | none | Update `isModelsLoading` | Matches new value |
| `SET_SCREEN_FEEDBACK` | UI feedback triggered/cleared | none | Update `screenFeedback` | Matches new value |

## 4) ML Dataset State Transition Catalog

| Transition | Trigger | Precondition | State Changes | Postcondition |
|---|---|---|---|---|
| `SET_DATASET_OPTIONS` | Manifest fetched | none | Replace `datasetOptions` | Options match fetched manifest |
| `SELECT_DATASET` | User selects dataset | `datasetOptions.length > 0` | Update `selectedDatasetId` | `selectedDatasetId` is within valid options or null |
| `CACHE_DATASET` | Dataset content fetched | `selectedDatasetId` is not null | Add entry to `datasetCache` | `datasetCache[datasetId]` is populated |
| `SET_LOADING_MANIFEST` | Manifest fetch starts/ends | none | Update `isLoadingManifest` | Matches new value |
| `SET_LOADING_DATASET` | Dataset fetch starts/ends | none | Update `isLoadingDataset` | Matches new value |
| `SET_MANIFEST_LOADED` | Manifest successfully loaded | none | `manifestLoaded = true` | `manifestLoaded === true` |
| `SET_ERROR` | Operation fails | none | Update `error` | `error` is the error message or null |

## 5) Selector Contracts

Selectors are pure functions from state to derived values. Every framework must implement these selectors with identical output for identical state input.

| Selector | Input | Output | Deterministic Rule |
|---|---|---|---|
| `selectMessages` | ChatState | `ChatMessage[]` | Returns `state.messages` reference |
| `selectIsSending` | ChatState | `boolean` | Returns `state.isSending` |
| `selectModelOptions` | ChatState | `ChatModelOption[]` | Returns `state.modelOptions` reference |
| `selectSelectedModelId` | ChatState | `string | null` | Returns `state.selectedModelId` |
| `selectCurrentHistoryValue` | ChatState | `string | null` | If `historyCursor` is null, returns null. Otherwise returns `inputHistory[historyCursor]` or null if out of bounds. |
| `selectDatasetOptions` | MlDatasetState | `MlDatasetOption[]` | Returns `state.datasetOptions` reference |
| `selectSelectedDataset` | MlDatasetState | `MlDatasetCacheEntry | null` | If `selectedDatasetId` is null or not in cache, returns null. Otherwise returns `datasetCache[selectedDatasetId]`. |
| `selectIsDataReady` | MlDatasetState | `boolean` | Returns `manifestLoaded && !isLoadingManifest && !isLoadingDataset` |

## 6) State Invariants

- INV-S01: `historyCursor` must always be null or within `[0, inputHistory.length - 1]`. If `inputHistory` is empty, `historyCursor` must be null.
- INV-S02: `isSending` must never be true while another `SET_SENDING_TRUE` transition is pending (no double-send).
- INV-S03: `selectedModelId`, when non-null, must be the `id` of an entry in `modelOptions` OR must have been explicitly set before model options loaded (preserved across fetches).
- INV-S04: `datasetCache` keys must always be valid dataset IDs that have appeared in `datasetOptions` at some point during the session.
- INV-S05: After `SET_MANIFEST_LOADED` transitions to true, it must never transition back to false within the same session (manifest is immutable once loaded).
- INV-S06: `isLoadingManifest` and `manifestLoaded` must never both be true simultaneously.
- INV-S07: Initial state for all fields must match the Initial Value column in Sections 1 and 2 exactly when the state manager is first created.

## 7) Framework Mapping Guidance

This section is informational -- it does not prescribe implementation but clarifies how the portable contract maps to known framework idioms.

| Portable Concept | React/Zustand | Vue/Pinia | Svelte | Angular/ngrx |
|---|---|---|---|---|
| State shape | Zustand store state | Pinia store state | Writable store | ngrx State interface |
| Transitions | Zustand actions | Pinia actions | Store update functions | ngrx reducers + actions |
| Selectors | Zustand selectors / useStore(selector) | Pinia getters / storeToRefs | Derived stores ($:) | ngrx selectors |
| Side effects | Zustand actions calling async | Pinia actions calling async | Effect functions | ngrx effects |

## 8) Acceptance Checklist

- [ ] All state fields have explicit types, nullability, and initial values.
- [ ] All transitions have preconditions and postconditions.
- [ ] Selectors are deterministic and side-effect free.
- [ ] State invariants are falsifiable assertions.
- [ ] Framework mapping is informational only -- no implementation prescribed.
- [ ] Entity types referenced here match `api.spec.md` Section 3 definitions exactly.
