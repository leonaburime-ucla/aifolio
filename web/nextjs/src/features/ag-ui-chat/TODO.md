# Copilot Chat TODOs

## Current Blockers

- [ ] AG-UI frontend tool-call execution is still unreliable on `/ag-ui` for Agentic Research actions.
  - Symptoms: dataset switch/clear/remove calls are intermittently ignored or fail despite tools being registered.
  - Investigation focus: backend `actions` emission vs CopilotKit tool dispatch vs frontend handler binding.

- [ ] `/ag-ui` chat messages are not persisting reliably across route changes.
  - Symptoms: conversation state can reset after navigating away and returning.
  - Investigation focus: provider mount scope, persistence filtering, and Copilot runtime hydration order.

## Notes

- Keep debug logging enabled while these are unresolved.
- Do not remove compatibility aliases for tool names/args until tool-call telemetry confirms stable execution.
