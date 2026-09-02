/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // SatQuery brand palette — dark space theme
        space: {
          950: "#050811",
          900: "#0a1020",
          800: "#0f1a30",
          700: "#162040",
          600: "#1e2d55",
        },
        satellite: {
          400: "#60c5ff",
          500: "#3aabff",
          600: "#0d8fe0",
        },
        terra: {
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a",
        },
        sar: {
          400: "#f59e0b",
          500: "#d97706",
        },
        change: {
          red: "#ef4444",
          green: "#22c55e",
          yellow: "#eab308",
        },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      backgroundImage: {
        "space-gradient": "linear-gradient(135deg, #050811 0%, #0a1020 50%, #0f1630 100%)",
        "card-gradient": "linear-gradient(135deg, rgba(22,32,64,0.8) 0%, rgba(15,26,48,0.9) 100%)",
        "satellite-glow": "radial-gradient(ellipse at center, rgba(58,171,255,0.15) 0%, transparent 70%)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow": "spin 4s linear infinite",
        "fade-in": "fadeIn 0.5s ease-in-out",
        "slide-up": "slideUp 0.4s ease-out",
        "scan-line": "scanLine 2s linear infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(20px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        scanLine: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
      },
      boxShadow: {
        "satellite": "0 0 20px rgba(58,171,255,0.2), 0 4px 6px rgba(0,0,0,0.3)",
        "satellite-lg": "0 0 40px rgba(58,171,255,0.3), 0 10px 25px rgba(0,0,0,0.4)",
        "inner-glow": "inset 0 0 20px rgba(58,171,255,0.1)",
      },
    },
  },
  plugins: [],
};
