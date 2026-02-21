# AI Chat (Sidebar) Spec

Spec ID: ai-chat
Version: 1.1.0
Status: Draft
Last updated: 2026-01-20

---

## 1. Overview

A sidebar AI chat experience with a scrollable message history and a bottom-anchored input bar (OpenAI/Claude style). The component must support message history recall via Arrow Up/Down and be usable as a standalone UI element across platforms.

---

## 2. Scope

In scope:
- Sidebar chat container UI
- Scrollable message history
- Bottom-anchored input bar
- Input history navigation (Arrow Up/Down)
- State + actions for message lifecycle

Out of scope:
- Attachments
- Voice input
- Rich text formatting
- Threading

---

## 3. Inputs / Outputs

Inputs:
- User input text
- Up/Down arrow navigation
- Orchestrator API responses (assistant messages)

Outputs:
- Rendered chat transcript
- Updated input value on history navigation
- Updated message list on submit + response

---

## 4. Requirements (Numbered for Traceability)

### 4.1 UI Requirements
- 1.1.1 The chat sidebar is fixed to the side of the viewport and vertically fills available height.
- 1.1.2 The message history region is scrollable and preserves order.
- 1.1.3 The chat input bar is pinned to the bottom of the sidebar.
- 1.1.4 The plus button is visible but disabled and shows tooltip text "Disabled for now" on hover or click.
- 1.1.5 The input placeholder reads "Ask anything".
- 1.1.6 The Send button triggers submit.

### 4.2 Interaction Requirements
- 1.2.1 Enter submits the current input.
- 1.2.2 Arrow Up cycles backward through input history.
- 1.2.3 Arrow Down cycles forward through input history and clears when past the newest entry.
- 1.2.4 Submitting an empty or whitespace-only message is ignored.
- 1.2.5 History navigation does not modify stored message history.

### 4.3 State Requirements
- 1.3.1 `messages: { id, role, content, createdAt }[]` stores user + assistant messages.
- 1.3.2 `inputHistory: string[]` stores submitted input values.
- 1.3.3 `historyCursor: number | null` tracks history navigation.
- 1.3.4 `isSending: boolean` tracks in-flight requests.

### 4.4 Orchestration Requirements
- 1.4.1 `submit` trims input and ignores empty submissions.
- 1.4.2 `submit` appends the user message to state.
- 1.4.3 `submit` persists the input to `inputHistory`.
- 1.4.4 `submit` sends the request to the AI API (placeholder until wired).
- 1.4.5 `handleHistory(direction)` returns the next input value for the given direction.
- 1.4.6 `resetHistoryCursor()` clears input history navigation state.

---

## 5. Acceptance Criteria

- 2.1.1 The sidebar renders a scrollable transcript area and pinned input bar.
- 2.1.2 Clicking Send or pressing Enter adds a user message to the transcript.
- 2.1.3 Arrow Up/Down updates the input value with prior entries.
- 2.1.4 Disabled plus button always shows tooltip on hover/click.
- 2.1.5 Empty submissions do not add messages or change history.

---

## 6. Edge Cases

- 3.1.1 Arrow Up with no history leaves input unchanged.
- 3.1.2 Arrow Down past the newest history entry clears the input.
- 3.1.3 Rapid submissions preserve order and do not lose messages.

---

## 7. Non-Goals

- 4.1.1 File uploads
- 4.1.2 Voice input
- 4.1.3 Markdown rendering

---

## 8. Test Mapping (Template)

| Requirement ID | Test ID | Notes |
| --- | --- | --- |
| 1.1.1 | TBD | Sidebar layout |
| 1.2.2 | TBD | History navigation |

---

## 9. Versioning & Changes

- 1.1.0 Initial sidebar chat spec with numbered requirements and acceptance criteria.
