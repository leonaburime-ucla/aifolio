# AI-Powered Multi-Module Dashboard - Project Specification

## Project Overview

Intelligent dashboard with AI agents that analyze data, make decisions, and dynamically route users through interfaces based on conversational input. The entire roadmap is treated as an Orc-BASH + Domain-Driven Design (DDD) exercise so every feature can be traced back to explicit contracts, orchestrated workflows, and bounded contexts.

## Architecture Approach (Orc-BASH + DDD)

- **Reference:** [The Orc-BASH Pattern – Orchestrated Architecture for Maximum Reusability](https://medium.com/@leonaburime/the-orc-bash-pattern-orchestrated-architecture-for-maximum-reusability-5d6b4734c9f6)
- **Spec-Driven Development:** Each capability originates in `project-specification.md`, then lands in shared types before any code exists.
- **Orc-BASH Layers:** Orchestrators → Business Logic → APIs with State + Hooks on the UI side. Contracts ensure one-way dependencies.
- **Bounded Contexts:** Every module (Crypto, Real Estate, etc.) is a distinct domain with its own ubiquitous language, aggregates, repositories, and event vocabulary. Cross-domain interactions go through orchestrators only.
- **Domain Contracts:** `DomainAgentSpec`, aggregate schemas, and view models document ubiquitous concepts so the UI, LangGraph, and FastAPI runtimes stay consistent.
- **Testing Story:** SHARP tests validate business rules per bounded context; SUIF guards the presentation boundary and hook orchestration.

---

## Technology Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui
- Zustand or React Context

### Backend
- Next.js API Routes
- LangChain (chains, tools, document loaders)
- LangGraph (agent workflows)
- LangSmith (monitoring, debugging)

### Database
- Supabase (PostgreSQL)
- pgvector (embeddings)
- Prisma (ORM)
- Supabase Auth

### AI
- OpenAI GPT-4 (reasoning)
- OpenAI text-embedding-3-small (embeddings)
- Redis/Upstash (caching, rate limiting)

### DevOps
- Vercel (deployment)
- GitHub (version control)
- Sentry (error tracking)
- PostHog (analytics)

---

## Testing

### Patterns
- **SHARP**: Complex components (agent logic, calculations, workflows)
  - https://medium.com/@leonaburime/the-sharp-pattern-a-test-first-paradigm-for-react-react-native-apps-with-ai-f1853df47390
- **SUIF**: Simple components (cards, charts, displays)
  - https://medium.com/@leonaburime/the-suif-pattern-a-simpler-pragmatic-sharp-variant-b222e731ad0b

### Coverage
- Unit tests: Individual functions, agent nodes, API wrappers
- Integration tests: API sequences, database ops, agent workflows
- E2E tests: Complete user flows, conversation-to-routing

### Mocking
- All external APIs (OpenAI, crypto, real estate APIs)
- Database queries (Prisma mocks)
- Time-dependent functions

---

## Shared Infrastructure

### Components
- Sidebar navigation (module switcher)
- Universal chat interface
- Base agent class (routing, common tools)
- Auth system

### Database Core
- Users
- Chat history (cross-module)
- User preferences
- Embeddings (pgvector)

---

## Project 1: Cryptocurrency Analyzer

### Features
- Portfolio management (track holdings, value, gains/losses)
- Real-time prices and market data
- Market analysis (trends, sentiment)
- Price alerts
- AI-generated investment insights

**Bounded Context Notes**
- Aggregate roots: `CryptoPortfolio`, `Holding`, `MarketInsight`.
- Repositories: `crypto_portfolios`, caches, embeddings.
- Ubiquitous language: holdings, deltas, market regimes, alerts.
- Domain services map directly to Orc-BASH Business Logic (`calculatePortfolioDelta`, `scoreMarketRegime`).

### Data Sources
- CoinGecko API (prices, market data, historical)
- CoinCap API (backup data source)
- News API (crypto news, sentiment)
- Reddit API (community sentiment)

### Agent Capabilities
- Intent classification (view portfolio, analyze, get recommendations)
- Multi-step workflows (fetch holdings → prices → calculate → insights)
- Dynamic routing decisions

### Routes
- `/crypto/portfolio` - Portfolio overview
- `/crypto/portfolio?filter=btc` - Filtered holdings
- `/crypto/analysis/[symbol]` - Coin analysis
- `/crypto/alerts` - Alert management
- `/crypto/markets` - Market overview

### Database Tables
- crypto_portfolios
- crypto_holdings
- crypto_transactions
- crypto_price_cache
- crypto_alerts
- crypto_analysis_reports
- crypto_embeddings (pgvector)

### Testing
- SHARP: Portfolio calculations, agent workflows, investment analysis
- SUIF: Price cards, charts, portfolio lists
- Mock all crypto APIs
- Test routing accuracy


## LangGraph Agent Architecture

### Base Agent
- Intent classification node
- Tool execution orchestration
- Dynamic routing logic
- State management
- Error handling

### Workflow Pattern
1. Receive user input
2. Classify intent
3. Execute tools (APIs, calculations, queries)
4. Synthesize results
5. Generate response
6. Determine routing (navigate, display, etc.)
7. Return structured response

### LangGraph Features
- Conditional edges (branch on intent)
- Loops (retry, refine queries)
- Parallel execution (multiple API calls)
- State management (conversation context)
- Human-in-the-loop (clarifications)

### LangSmith
- Trace agent execution
- Monitor performance and costs
- Debug workflows
- Track accuracy

---

## API Structure

### Endpoints
```
/api/v1/chat                     # Generic chat (routes to module agent)
```

### Agent Response Format
```json
{
  "message": "Agent response text",
  "action": "navigate",
  "route": "/crypto/portfolio?filter=btc",
  "data": { ... }
}
```

---

## Dynamic Routing Examples

### Crypto
- "Show my Bitcoin" → `/crypto/portfolio?filter=btc`
- "Analyze Ethereum trends" → `/crypto/analysis/ethereum`
- "Alert me if BTC hits $50k" → `/crypto/alerts/new?symbol=btc&price=50000`


## Security

- JWT authentication
- Environment variables for API keys
- Rate limiting
- Input sanitization
- HTTPS only
- Prompt injection protection

---

## Performance

- Cache external API responses
- Redis for sessions
- Database indexing
- Code splitting
- Parallel tool execution
- CDN for static assets

---

## Monitoring

- Sentry (errors)
- Vercel Analytics (performance)
- PostHog (user behavior)
- LangSmith (agent execution)

---

## Cost Estimate

- Supabase: $0-25/month
- Vercel: $0-20/month
- OpenAI API: $10-100/month
- External APIs: $0-50/month
- Monitoring: $0-26/month

**Total: $10-200/month depending on usage**
