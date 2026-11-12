/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0810",
          900: "#0f0c16",
          850: "#14101c",
          800: "#1a1524",
          700: "#241d31",
          600: "#332942",
        },
        brand: {
          50: "#f7f0ff",
          100: "#efe0ff",
          300: "#cfa6ff",
          400: "#b57bff",
          500: "#9a4dfc",
          600: "#8430e0",
          700: "#6c1fb8",
          900: "#3b0f68",
        },
      },
      fontFamily: {
        sans: ["Inter", "Manrope", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(154,77,252,0.25), 0 18px 60px -24px rgba(154,77,252,0.55)",
        card: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 20px 40px -32px rgba(0,0,0,0.9)",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-in": "fade-in .25s ease-out",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
