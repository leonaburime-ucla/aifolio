/** @type {import('tailwindcss').Config} */
const config = {
  content: [
    "./src/**/*.{ts,tsx,js,jsx,mdx}",
    "../src/ui/**/*.{ts,tsx,js,jsx,mdx}",
    "../src/ui-hooks/**/*.{ts,tsx,js,jsx,mdx}",
    "../src/logic/**/*.{ts,tsx,js,jsx,mdx}",
    "../src/state-management/**/*.{ts,tsx,js,jsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};

module.exports = config;
