"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

/* ──────────────────────────────────────────────────────────────────────────
   SatQuery theme engine.
   - Preference is one of: "system" | "light" | "dark"  (persisted to localStorage)
   - resolvedTheme is the concrete "light" | "dark" actually applied
   - Follows OS preference by default; a manual toggle sets an explicit override
   - The no-flash <script> in layout.tsx sets data-theme before first paint, so
     this provider only *syncs* React state to what's already on <html>.
   ────────────────────────────────────────────────────────────────────────── */

export type ThemePref = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "sq-theme";
const META_COLOR: Record<ResolvedTheme, string> = {
  light: "#E0D1C0",
  dark: "#1C1F20",
};

type ThemeCtx = {
  theme: ThemePref;
  resolvedTheme: ResolvedTheme;
  setTheme: (t: ThemePref) => void;
  toggle: () => void;
};

const ThemeContext = createContext<ThemeCtx | null>(null);

function systemPref(): ResolvedTheme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function resolve(pref: ThemePref): ResolvedTheme {
  return pref === "system" ? systemPref() : pref;
}

function apply(resolved: ResolvedTheme) {
  const root = document.documentElement;
  root.setAttribute("data-theme", resolved);
  root.style.colorScheme = resolved;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", META_COLOR[resolved]);
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Start from a safe default; the mount effect reconciles with storage + DOM.
  const [theme, setThemeState] = useState<ThemePref>("system");
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>("dark");

  // On mount: read stored preference and sync state to what's on <html>.
  useEffect(() => {
    let stored: ThemePref = "system";
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw === "light" || raw === "dark" || raw === "system") stored = raw;
    } catch {
      /* private mode / storage disabled */
    }
    const r = resolve(stored);
    setThemeState(stored);
    setResolvedTheme(r);
    apply(r);
  }, []);

  // While following the system, react to OS-level theme changes live.
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      const r = systemPref();
      setResolvedTheme(r);
      apply(r);
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = useCallback((pref: ThemePref) => {
    try {
      localStorage.setItem(STORAGE_KEY, pref);
    } catch {
      /* ignore */
    }
    const r = resolve(pref);
    setThemeState(pref);
    setResolvedTheme(r);
    apply(r);
  }, []);

  // Quick toggle flips to the opposite of what's currently shown (explicit).
  const toggle = useCallback(() => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  }, [resolvedTheme, setTheme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeCtx {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within <ThemeProvider>");
  return ctx;
}

/** True only after client mount: use to avoid rendering theme-specific UI on the server. */
export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}
