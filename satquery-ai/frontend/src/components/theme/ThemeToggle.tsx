"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme, useMounted, type ThemePref } from "./ThemeProvider";

/* ──────────────────────────────────────────────────────────────────────────
   ThemeToggle — compact sun/moon button for the app chrome.
   Clicking sets an explicit light/dark override; a small dot marks when the
   app is still following the system preference.
   ────────────────────────────────────────────────────────────────────────── */
export default function ThemeToggle({ className = "" }: { className?: string }) {
  const { resolvedTheme, theme, toggle } = useTheme();
  const mounted = useMounted();

  // Render a neutral placeholder pre-mount to avoid a hydration mismatch.
  if (!mounted) {
    return (
      <span
        className={`inline-flex items-center justify-center w-10 h-10 rounded-lg ${className}`}
        style={{ border: "1px solid var(--border)" }}
        aria-hidden
      />
    );
  }

  const isDark = resolvedTheme === "dark";
  const followingSystem = theme === "system";

  return (
    <button
      onClick={toggle}
      className={`group relative inline-flex items-center justify-center w-10 h-10 rounded-lg no-tap transition-colors ${className}`}
      style={{ border: "1px solid var(--border)", color: "var(--text-muted)" }}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={
        followingSystem
          ? `Following system (${resolvedTheme}) — click to override`
          : `Theme: ${resolvedTheme} — click to switch`
      }
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={isDark ? "moon" : "sun"}
          initial={{ opacity: 0, rotate: -35, scale: 0.7 }}
          animate={{ opacity: 1, rotate: 0, scale: 1 }}
          exit={{ opacity: 0, rotate: 35, scale: 0.7 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
          className="inline-flex group-hover:text-[var(--text-primary)]"
          style={{ color: "inherit" }}
        >
          {isDark ? <Moon className="w-[18px] h-[18px]" /> : <Sun className="w-[18px] h-[18px]" />}
        </motion.span>
      </AnimatePresence>

      {followingSystem && (
        <span
          className="absolute bottom-1.5 right-1.5 w-1.5 h-1.5 rounded-full"
          style={{ background: "var(--accent)" }}
        />
      )}
    </button>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
   ThemeSegment — full System / Light / Dark control (for the Settings page).
   ────────────────────────────────────────────────────────────────────────── */
const SEGMENTS: { value: ThemePref; label: string; Icon: typeof Sun }[] = [
  { value: "system", label: "System", Icon: Monitor },
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
];

export function ThemeSegment() {
  const { theme, setTheme } = useTheme();
  const mounted = useMounted();
  const active = mounted ? theme : "system";

  return (
    <div
      className="inline-flex items-center gap-1 p-1 rounded-xl"
      style={{ background: "var(--bg-raised)", border: "1px solid var(--border)" }}
      role="radiogroup"
      aria-label="Theme"
    >
      {SEGMENTS.map(({ value, label, Icon }) => {
        const isActive = active === value;
        return (
          <button
            key={value}
            role="radio"
            aria-checked={isActive}
            onClick={() => setTheme(value)}
            className="relative flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium no-tap transition-colors"
            style={{ color: isActive ? "var(--accent-contrast)" : "var(--text-muted)" }}
          >
            {isActive && (
              <motion.span
                layoutId="theme-seg"
                className="absolute inset-0 rounded-lg"
                style={{ background: "var(--accent)" }}
                transition={{ type: "spring", stiffness: 400, damping: 32 }}
              />
            )}
            <Icon className="w-4 h-4 relative z-10" />
            <span className="relative z-10">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
