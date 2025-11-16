import type { ProjectSpec } from "./types";

export const projectSpec: ProjectSpec = {
  name: "AI-Powered Multi-Module Dashboard",
  description:
    "Intelligent dashboard with AI agents that analyze data, make decisions, and dynamically route users through conversational input.",
  stack: [
    "Next.js 14 App Router",
    "TypeScript",
    "Tailwind CSS v4",
    "shadcn/ui",
    "Zustand",
    "LangChain",
    "LangGraph",
    "LangSmith",
    "Supabase + pgvector",
    "Prisma",
    "OpenAI + Upstash Redis",
  ],
  patternStrategy:
    "Complex decisioning and data workflows follow SHARP while display-first widgets use SUIF.",
  modules: [
    {
      id: "crypto",
      title: "Cryptocurrency Analyzer",
      summary:
        "Portfolio-aware assistant that fetches holdings, normalizes pricing data, and produces AI insight cards for traders.",
      status: "in-progress",
      entryRoute: "/crypto/portfolio",
      designPattern: "sharp",
      routes: [
        {
          path: "/crypto/portfolio",
          title: "Portfolio overview",
          description: "Summaries positions, P&L, allocation, and agent guidance.",
          discipline: "suif",
        },
        {
          path: "/crypto/portfolio?filter=btc",
          title: "Filtered holdings",
          description: "Focus view for a single asset or tag.",
          discipline: "suif",
        },
        {
          path: "/crypto/analysis/[symbol]",
          title: "Coin deep dive",
          description: "Aggregates market, news, and sentiment for a coin.",
          pattern: "^/crypto/analysis/[a-z0-9-]+$",
          discipline: "sharp",
        },
        {
          path: "/crypto/alerts",
          title: "Alert management",
          description: "CRUD for price and sentiment alerts.",
          discipline: "sharp",
        },
        {
          path: "/crypto/markets",
          title: "Market overview",
          description: "Heatmap of top movers, dominance, and macro indicators.",
          discipline: "suif",
        },
      ],
      dataSources: [
        {
          id: "coingecko",
          name: "CoinGecko API",
          description: "Primary prices, market caps, historical candles.",
          docsUrl: "https://www.coingecko.com/en/api",
          type: "api",
          critical: true,
        },
        {
          id: "coincap",
          name: "CoinCap API",
          description: "Backup pricing and exchange metrics.",
          docsUrl: "https://docs.coincap.io/",
          type: "api",
          critical: false,
        },
        {
          id: "news",
          name: "News API",
          description: "Editorial headlines + sentiment.",
          docsUrl: "https://newsapi.org/",
          type: "api",
          critical: true,
        },
        {
          id: "reddit",
          name: "Reddit API",
          description: "Community mentions and sentiment proxy.",
          docsUrl: "https://www.reddit.com/dev/api",
          type: "api",
          critical: false,
        },
      ],
      tables: [
        {
          name: "crypto_portfolios",
          description: "User level snapshots of preferred allocations.",
          columns: ["id", "user_id", "label", "base_currency", "created_at"],
          tags: ["core"],
        },
        {
          name: "crypto_holdings",
          description: "Individual holdings tied to a portfolio.",
          columns: ["id", "portfolio_id", "symbol", "quantity", "cost_basis"],
          tags: ["core"],
        },
        {
          name: "crypto_transactions",
          description: "Trade history enabling realized P&L.",
          columns: ["id", "portfolio_id", "symbol", "side", "qty", "executed_at"],
          tags: ["core"],
        },
        {
          name: "crypto_price_cache",
          description: "Redis + Postgres hybrid cache for price lookups.",
          columns: ["symbol", "payload", "fetched_at"],
          tags: ["cache"],
        },
        {
          name: "crypto_alerts",
          description: "User-configurable alerts.",
          columns: ["id", "user_id", "symbol", "direction", "threshold", "active"],
          tags: ["core"],
        },
        {
          name: "crypto_embeddings",
          description: "pgvector store for semantic recall.",
          columns: ["id", "source_type", "source_id", "embedding"],
          tags: ["embedding"],
        },
      ],
      features: [
        {
          id: "portfolio-tracking",
          title: "Portfolio management",
          description: "Track holdings, allocations, and total value in real time.",
          routes: ["/crypto/portfolio"],
          tests: {
            sharp: ["portfolio math stays deterministic"],
            suif: ["render cards + charts"],
          },
        },
        {
          id: "market-analysis",
          title: "Market analysis",
          description: "Trend detection w/ news + social sentiment blending.",
          routes: ["/crypto/analysis/[symbol]", "/crypto/markets"],
          tests: {
            sharp: ["agent chooses best tool chain", "sentiment mixing"],
          },
        },
        {
          id: "price-alerts",
          title: "Price alerts",
          description: "Notifications when guardrails hit.",
          routes: ["/crypto/alerts"],
          tests: {
            sharp: ["threshold evaluation"],
            suif: ["alert surface states"]
          },
        },
      ],
      agentCapabilities: [
        {
          id: "crypto-intents",
          description: "Intent classification for holdings, prices, alerts, analysis.",
          workflow: ["classify", "hydrate data", "respond"],
        },
        {
          id: "crypto-routing",
          description: "Dynamic routing to module routes based on conversation context.",
          workflow: ["evaluate keywords", "rank best route", "emit action"],
        },
      ],
    },
    {
      id: "real-estate",
      title: "Real Estate Market Intelligence",
      summary:
        "Natural-language property search with enriched neighborhood, school, and investment scoring.",
      status: "planned",
      entryRoute: "/real-estate/search",
      designPattern: "sharp",
      routes: [
        {
          path: "/real-estate/search",
          title: "Property search",
          description: "Filterable search results for NL queries.",
          discipline: "sharp",
        },
        {
          path: "/real-estate/property/[id]",
          title: "Property details",
          description: "Cross-source property dossier.",
          pattern: "^/real-estate/property/[a-z0-9-]+$",
          discipline: "suif",
        },
        {
          path: "/real-estate/neighborhoods/[id]",
          title: "Neighborhood analysis",
          description: "Demographics, crime, schools, lifestyle.",
          discipline: "sharp",
        },
        {
          path: "/real-estate/compare",
          title: "Property comparison",
          description: "Multi-property deltas with AI commentary.",
          discipline: "suif",
        },
        {
          path: "/real-estate/analysis/investment",
          title: "Investment calculator",
          description: "ROI, cap rate, and cash flow modeling.",
          discipline: "sharp",
        },
        {
          path: "/real-estate/markets/[city]",
          title: "Market trends",
          description: "City level macro stats.",
          discipline: "suif",
        },
      ],
      dataSources: [
        {
          id: "realtor",
          name: "Realtor.com API",
          description: "Listings and valuations.",
          docsUrl: "https://www.realtor.com/research/data/",
          type: "api",
          critical: true,
        },
        {
          id: "walkscore",
          name: "Walk Score API",
          description: "Lifestyle + commute metrics.",
          docsUrl: "https://www.walkscore.com/professional/api.php",
          type: "api",
          critical: false,
        },
        {
          id: "greatschools",
          name: "GreatSchools API",
          description: "School ratings and metadata.",
          docsUrl: "https://apidocs.greatschools.org/",
          type: "api",
          critical: false,
        },
        {
          id: "census",
          name: "US Census API",
          description: "Demographics + incomes.",
          docsUrl: "https://www.census.gov/data/developers/data-sets.html",
          type: "api",
          critical: true,
        },
        {
          id: "crime",
          name: "Crime data APIs",
          description: "Local crime stats.",
          docsUrl: "https://crime-data-explorer.app.cloud.gov/pages/docApi",
          type: "api",
          critical: false,
        },
        {
          id: "maps",
          name: "Google Maps Platform",
          description: "Commute, distances, route planning.",
          docsUrl: "https://developers.google.com/maps/documentation",
          type: "api",
          critical: true,
        },
      ],
      tables: [
        {
          name: "saved_properties",
          description: "User-saved favorites across sources.",
          columns: ["id", "user_id", "external_id", "payload"],
          tags: ["core"],
        },
        {
          name: "property_cache",
          description: "Cached listing responses.",
          columns: ["id", "payload", "fetched_at"],
          tags: ["cache"],
        },
        {
          name: "neighborhood_cache",
          description: "Neighborhood metadata + scores.",
          columns: ["id", "payload", "fetched_at"],
          tags: ["cache"],
        },
        {
          name: "market_trends_cache",
          description: "Macro stats reused across users.",
          columns: ["city", "payload", "fetched_at"],
          tags: ["cache"],
        },
        {
          name: "investment_analyses",
          description: "Persisted ROI, cash-flow calculations.",
          columns: ["id", "user_id", "inputs", "outputs", "created_at"],
          tags: ["core"],
        },
        {
          name: "saved_searches",
          description: "Reusable natural-language searches.",
          columns: ["id", "user_id", "query", "filters", "created_at"],
          tags: ["core"],
        },
        {
          name: "property_embeddings",
          description: "pgvector embeddings for property + doc search.",
          columns: ["id", "source_id", "embedding"],
          tags: ["embedding"],
        },
        {
          name: "neighborhood_embeddings",
          description: "Vector store for area insights.",
          columns: ["id", "source_id", "embedding"],
          tags: ["embedding"],
        },
      ],
      features: [
        {
          id: "property-search",
          title: "Property search",
          description: "NL prompt to filters + ranking.",
          routes: ["/real-estate/search"],
          tests: {
            sharp: ["intent parsing", "filter builder"],
          },
        },
        {
          id: "neighborhood-analysis",
          title: "Neighborhood analysis",
          description: "Aggregate data + AI summary.",
          routes: ["/real-estate/neighborhoods/[id]"],
          tests: {
            sharp: ["scoring weights"],
            suif: ["scorecard visuals"],
          },
        },
        {
          id: "investment-calculator",
          title: "Investment calculator",
          description: "ROI, IRR, and stress scenarios.",
          routes: ["/real-estate/analysis/investment"],
          tests: {
            sharp: ["cashflow math"],
          },
        },
      ],
      agentCapabilities: [
        {
          id: "re-intents",
          description: "Parse multi-constraint property briefs.",
          workflow: ["parse features", "call data APIs", "score"],
        },
        {
          id: "re-routing",
          description: "Route to search, neighborhood, compare, or investment screens.",
          workflow: ["classify", "rank routes", "respond"],
        },
      ],
    },
  ],
  sharedComponents: [
    {
      id: "sidebar",
      title: "Sidebar navigation",
      description: "Module switcher + nav tree.",
    },
    {
      id: "chat",
      title: "Universal chat",
      description: "Entry point for every module agent.",
    },
    {
      id: "base-agent",
      title: "Base agent",
      description: "Intent classification, routing, error handling.",
    },
    {
      id: "auth",
      title: "Supabase Auth",
      description: "Shared auth + session context.",
    },
  ],
  testing: {
    sharpFocus: [
      "Portfolio math & ROI calculators",
      "Agent workflows per module",
      "Routing logic from chat to routes",
    ],
    suifFocus: [
      "Cards, charts, comparison tables",
      "Sidebar and navigation state",
    ],
    coverageGoals: [
      ">=80% statements on SHARP targets",
      ">=60% on UI widgets",
      "E2E smoke for conversation-to-route",
    ],
    mocking: [
      "OpenAI + LangSmith",
      "Crypto + Real estate APIs",
      "Prisma DB calls",
      "Timers and caching",
    ],
  },
  security: [
    "JWT auth via Supabase",
    "Rate limiting + Upstash Redis",
    "Prompt injection guardrails",
    "HTTPS-only deployments",
  ],
  monitoring: [
    "Sentry for errors",
    "PostHog for behavior",
    "LangSmith traces",
    "Vercel Analytics",
  ],
};
