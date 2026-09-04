/** @type {import('tailwindcss').Config} */
module.exports = {
  // Theme is switched by [data-theme="dark"] on <html> (set pre-paint in layout.tsx)
  darkMode: ['selector', '[data-theme="dark"]'],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // ── Semantic color tokens (all backed by CSS vars → theme-aware) ──────
      colors: {
        // Surfaces
        base:     "var(--bg-base)",
        surface:  "var(--bg-surface)",
        raised:   "var(--bg-raised)",
        inset:    "var(--bg-inset)",
        elevated: "var(--bg-elevated)",

        // Text  (use as text-ink / text-ink-muted / text-ink-faint)
        ink: {
          DEFAULT: "var(--text-primary)",
          soft:    "var(--text-secondary)",
          muted:   "var(--text-muted)",
          faint:   "var(--text-faint)",
        },

        // Lines
        line: {
          DEFAULT: "var(--border)",
          strong:  "var(--border-strong)",
        },
        border: "var(--border)", // legacy key kept

        // Interactive accent — spectral indigo
        accent: {
          DEFAULT: "var(--accent)",
          hover:   "var(--accent-hover)",
          active:  "var(--accent-active)",
          contrast:"var(--accent-contrast)",
          soft:    "var(--accent-soft)",
          border:  "var(--accent-border)",
        },
        primary: "var(--accent)", // sub-page alias (bg-primary / text-primary)

        // DATA palette — meaningful sensor / land-cover encodings
        veg:    { DEFAULT: "var(--veg)",    soft: "var(--veg-soft)"    },
        sar:    { DEFAULT: "var(--sar)",    soft: "var(--sar-soft)"    },
        water:  { DEFAULT: "var(--water)",  soft: "var(--water-soft)"  },
        bare:   { DEFAULT: "var(--bare)",   soft: "var(--bare-soft)"   },
        change: { DEFAULT: "var(--change)", soft: "var(--change-soft)" },

        // Status
        success: "var(--success)",
        warning: "var(--warning)",
        danger:  "var(--danger)",

        // ── Legacy palette remaps ─────────────────────────────────────────
        // Old class names keep working AND become theme-aware for free.
        space: {
          950: "var(--bg-base)",
          900: "var(--bg-surface)",
          800: "var(--bg-raised)",
          700: "var(--border-strong)",
          600: "var(--text-faint)",
        },
        satellite: {
          300: "var(--accent-hover)",
          400: "var(--accent-hover)",
          500: "var(--accent)",
          600: "var(--accent-active)",
          700: "var(--accent-active)",
        },
        terra:     { 400: "var(--veg)",    500: "var(--veg)",  600: "var(--veg)" },
        elevated2: "var(--bg-raised)",
      },

      // ── Typography ────────────────────────────────────────────────────────
      // IBM Plex Sans (humanist, technical heritage) + IBM Plex Mono (data only)
      fontFamily: {
        sans: ["var(--font-plex-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      letterSpacing: {
        tightest: "-0.03em",
      },

      // ── Background images ─────────────────────────────────────────────────
      backgroundImage: {
        "accent-glow":   "radial-gradient(ellipse at center, var(--accent-soft) 0%, transparent 70%)",
        "spectral-band": "linear-gradient(90deg, var(--veg), var(--water), var(--accent), var(--sar), var(--bare))",
      },

      // ── Shadows (token-backed → theme-aware) ───────────────────────────────
      boxShadow: {
        sm:     "var(--shadow-sm)",
        DEFAULT:"var(--shadow-md)",
        md:     "var(--shadow-md)",
        lg:     "var(--shadow-lg)",
        ring:   "var(--ring)",
      },

      // ── Animations ────────────────────────────────────────────────────────
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "spin-slow":  "spin 4s linear infinite",
        "fade-in":    "fadeIn 0.3s ease-out both",
        "slide-up":   "slideUp 0.3s ease-out both",
        "slide-in":   "slideIn 0.25s ease-out both",
        "scan-line":  "scanLine 2s linear infinite",
      },
      keyframes: {
        fadeIn:   { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideUp:  { "0%": { transform: "translateY(12px)", opacity: "0" }, "100%": { transform: "translateY(0)", opacity: "1" } },
        slideIn:  { "0%": { transform: "translateX(-12px)", opacity: "0" }, "100%": { transform: "translateX(0)", opacity: "1" } },
        scanLine: { "0%": { transform: "translateY(-100%)" }, "100%": { transform: "translateY(500%)" } },
      },

      screens: { xs: "375px" },
    },
  },
  plugins: [],
};
