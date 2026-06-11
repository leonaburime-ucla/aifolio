export default defineNuxtConfig({
  compatibilityDate: "2025-01-01",
  devtools: { enabled: true },

  srcDir: "src/",

  modules: ["@pinia/nuxt", "@nuxtjs/tailwindcss"],

  tailwindcss: {
    cssPath: "~/assets/css/globals.css",
  },

  app: {
    head: {
      title: "AIfolio",
      meta: [
        { name: "description", content: "AIfolio — AI-driven portfolio dashboard" },
      ],
      link: [
        {
          rel: "stylesheet",
          href: "https://fonts.googleapis.com/css2?family=Geist:wght@100..900&family=Geist+Mono:wght@100..900&display=swap",
        },
      ],
    },
  },

  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.NUXT_PUBLIC_API_BASE_URL || "http://localhost:8000",
      debugEffects: process.env.NUXT_PUBLIC_DEBUG_EFFECTS === "1",
    },
  },
});
