# AI Feature (Orc-BASH)

Purpose: chat sidebar, intent routing, tool orchestration.

- logic/api/react: implementation folders split by concern
- /react/orchestrators: React-bound orchestration hooks
- /react/hooks: React hooks
- /react/state: React state adapters and stores
- /react/views: React UI components
- __tests__: mirrored test tree for implementation surfaces

Versioned specs live in `ADS-project-knowledge/specs/001-multi-framework-frontend-core/nextjs-feature-specs/ai-chat`.
Reusable chat contracts live in `@aifolio/contracts/entities/chat`; React-only types are colocated with their React modules.
