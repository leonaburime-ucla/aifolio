# AG-UI Backend Slice

This module is now split so AG-UI can evolve independently from the rest of `server.py`.

## Files

- `ai/python/agui.py`: AG-UI orchestration + provider adapter boundary
- `ai/python/server.py`: route registration only (`POST /agui` delegates to `create_agui_stream_response`)

## Current flow

1. Frontend Copilot sends requests to `POST /api/copilotkit`.
2. Next.js Copilot runtime forwards to backend `POST /agui`.
3. `server.py` calls `create_agui_stream_response(payload)`.
4. `agui.py`:
   - validates `RunAgentInput`
   - emits AG-UI lifecycle events (`RUN_STARTED -> TEXT_MESSAGE_* -> RUN_FINISHED`)
   - on failure emits only `RUN_ERROR` and stops

## Provider abstraction

`agui.py` defines:

- `ChatProvider` protocol (interface)
- `LangChainGeminiProvider` default implementation (uses existing `run_chat`)

This lets you replace Gemini with other providers without changing the route contract.

## Add another provider

Implement:

```python
class OpenAIProvider:
    def generate(self, payload: dict[str, Any]) -> str:
        ...
```

Then inject it:

```python
return create_agui_stream_response(payload, provider=OpenAIProvider())
```

## Why this structure

- Keeps AG-UI as a vertical slice with its own orchestration boundary.
- Keeps route code thin and stable.
- Makes multi-provider support an adapter concern instead of route logic.
