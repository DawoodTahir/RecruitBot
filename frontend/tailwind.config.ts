import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"]
      },
      colors: {
        brand: {
          50: "#eff7ff",
          100: "#d6ebff",
          200: "#aed8ff",
          300: "#84c4ff",
          400: "#5cb1ff",
          500: "#329dff",
          600: "#1e7bd4",
          700: "#145aa0",
          800: "#0b3a6c",
          900: "#041c33"
        }
      },
      boxShadow: {
        elevated: "0 25px 45px -20px rgba(15, 23, 42, 0.6)"
      }
    }
  },
  plugins: []
};

export default config;

