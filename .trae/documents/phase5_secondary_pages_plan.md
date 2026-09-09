# Phase 5 — Secondary Pages and Navigation Implementation Plan

## Repository Research

### Current State Summary

**Navigation (Already Implemented Correctly):**
- [Sidebar.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/layout/Sidebar.tsx) — Contains all 5 nav items with correct hrefs (`/`, `/history`, `/datasets`, `/reports`, `/settings`), icons, and active-route highlighting via `usePathname`.
- [MobileBottomNav.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/layout/MobileBottomNav.tsx) — 5-tab bottom nav for mobile with correct routes and active state.
- [MobileHeader.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/layout/MobileHeader.tsx) — Slide-over drawer nav for mobile with all 5 items + active highlighting.
- [Header.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/layout/Header.tsx) — Desktop top header, shows correct page titles from `PAGE_TITLES` map for all 5 routes.
- [AppShell.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/layout/AppShell.tsx) — Wraps sidebar + header + mobile navs + main content. Used by RootLayout, so all pages inherit the shell.

**Routing:**
- All 5 Next.js App Router pages exist at correct paths: `/`, `/history`, `/datasets`, `/reports`, `/settings`. No duplicate routes.

**Design System (Dashboard Reference Palette — from page.tsx):**
- Page bg: `#0D1117`
- Card bg: `#161B22`
- Raised/hover bg: `#1C2333`
- Border: `rgba(48,54,61,0.8)` (subtle dividers `0.6`)
- Primary blue: `#388BFD`, light blue: `#58A6FF`
- Primary text: `#E6EDF3`, muted: `#8B949E`, faint: `#6E7681`, divider: `#30363D`
- Success: `#3FB950`, Error: `#F85149`, Amber: `#D29922`, Violet: `#A371F7`
- Cards use `DashCard` pattern: `rounded-xl` (12px), header border, consistent padding

### Identified Issues in Secondary Pages

1. **Generic Tailwind grays instead of dashboard palette** — All 4 sub-pages use `text-gray-500`, `text-gray-400`, `text-gray-300`, `text-gray-100` etc. These map to Tailwind's default grays, NOT the SatQuery palette.

2. **CSS class vs inline mismatch** — Dashboard uses explicit `style={{}}` or `text-[#XXX]` arbitrary values for exact colors. Sub-pages rely on Tailwind's default `text-gray-*` utility classes which render differently.

3. **Border-radius mismatch** — `globals.css` defines `.sq-card` (alias `.card`) with `border-radius: 10px`, but dashboard `DashCard` uses `rounded-xl` = 12px.

4. **History page uses MOCK_HISTORY instead of real localStorage data** — [SessionHistory.tsx](file:///d:/PROJECTS/SIH2026/satquery-ai/frontend/src/components/SessionHistory.tsx) has real localStorage-backed data (`STORAGE_KEY = "satquery_history"`). The history page should reuse the same data source, not hardcoded mocks.

5. **Minor styling variations** — Inputs, buttons, badges use slightly different Tailwind classes vs dashboard's exact hex colors.

---

## Files and Modules

### Files to Modify

1. **`frontend/src/app/history/page.tsx`** — Replace MOCK_HISTORY with localStorage data (reuse `STORAGE_KEY` / types from SessionHistory.tsx). Replace all generic Tailwind gray color classes with exact dashboard hex codes. Align card/header styling with the `DashCard` pattern. Add a fallback section when localStorage is empty. Use same task badge colors as SessionHistory.tsx.

2. **`frontend/src/app/datasets/page.tsx`** — Update all `text-gray-*` / `bg-*` Tailwind classes to match dashboard palette hex codes. Align card borders, buttons, search input styling to match dashboard pattern. Preserve existing DATASETS array (task says "do not invent dataset records" — existing mock records are fine).

3. **`frontend/src/app/reports/page.tsx`** — Update all color classes to dashboard palette. Align card/badge/button styling. Preserve existing REPORTS data and existing download/view links.

4. **`frontend/src/app/settings/page.tsx`** — Update all color classes to dashboard palette. Align tab buttons, inputs, cards to match dashboard exact hex styling. Preserve all 4 tabs and their state.

5. **`frontend/src/app/globals.css`** — Fix `.sq-card` (`.card`) border-radius from `10px` to `12px` to match dashboard `rounded-xl`.

### Files NOT Modified (Already Correct)
- Sidebar.tsx, MobileBottomNav.tsx, MobileHeader.tsx, Header.tsx, AppShell.tsx — navigation is correct.
- layout.tsx, page.tsx (dashboard) — no changes needed.
- SessionHistory.tsx component itself — keep intact, only reuse its data pattern in history page.
- No backend modifications.

---

## Implementation Steps

### Step 1: Fix global card border-radius
- In `globals.css`, update `.sq-card` `border-radius` from `10px` to `12px` (matches `rounded-xl` used by dashboard `DashCard`).

### Step 2: Refine Execution History Page (highest priority)
- Import `HistoryEntry` type and `STORAGE_KEY` from `@/components/SessionHistory` (or replicate the storage read pattern inline).
- Replace `MOCK_HISTORY` with a `useEffect` that reads from `localStorage.getItem(STORAGE_KEY)` on mount.
- Build table rows from actual entries, mapping fields:
  - `timestamp` → format as date/time
  - `query` → query column
  - `task` → tasks/badge column
  - `confidence` → confidence column
  - `session_id` → display in metadata
- Reuse the exact task badge color map from `SessionHistory.tsx` (`TASK_COLOR` record with `_VIOLET/SKY/EMERALD/AMBER/ORANGE/ROSE`).
- Replace all `text-gray-*` / `bg-gray-*` / colors with exact dashboard hex codes using `text-[#XXX]`, `bg-[#XXX]`, or `style={{}}`.
- Update buttons, inputs, selects to use exact palette (`#161B22` bg, `rgba(48,54,61,0.8)` border, `#8B949E` placeholder etc.).
- Add a friendly empty state when no history entries exist (consistent with dashboard empty states).

### Step 3: Refine Datasets Page
- Replace all `text-gray-*` classes with exact palette hex codes.
- Update card, input, button styling to match dashboard (use `#161B22` bg for cards, `rgba(48,54,61,0.8)` borders, `rounded-xl`, correct text colors).
- Update `TYPE_COLOR` record to match the dashboard's saturated color palette (`bg-XXX-950/50` → use `rgba(X,X,X,0.08)` style or explicit hex bg with matching border + text colors from dashboard).
- Keep existing DATASETS array, upload handlers, upload dropzone JSX intact.

### Step 4: Refine Reports Page
- Replace all `text-gray-*` classes with exact palette hex codes.
- Align card styling with `DashCard` pattern (correct border-radius, bg, text colors).
- Update `TASK_COLOR` map to use dashboard-consistent colors.
- Keep existing `NEXT_PUBLIC_API_URL` logic, `href` links for View/Download unchanged.
- Keep existing REPORTS array intact.

### Step 5: Refine Settings Page
- Replace all `text-gray-*` / `bg-elevated/60` generic classes with exact hex codes.
- Align the left tab-nav card and right content card to match `DashCard` visual style.
- Update input fields, tab buttons, toggle rows to use exact palette colors.
- Keep all 4 tabs (profile, application, models, security) state logic, toggle state, and form inputs intact.
- Preserve API key display, password change inputs, save buttons.

### Step 6: Run Validation Checks
- Build the project: `npm run build` in `frontend/` directory.
- Check for TypeScript errors via `npx tsc --noEmit` or via `GetDiagnostics`.
- Test navigation manually via dev server (if running):
  - Click each sidebar nav item → confirms route, highlight, and header title.
  - Test mobile nav (bottom tabs) and mobile slide-over drawer.
- Verify no console errors (using browser devtools or running tests if available).

---

## Dependencies and Considerations

- **No new dependencies.** Using existing React hooks, lucide-react icons, framer-motion (already in pages).
- **Data source:** SessionHistory's localStorage key `"satquery_history"` is the source of truth for the history page. Will render empty if user hasn't run analyses yet.
- **Preservation constraint:** All existing functional handlers (file uploads on datasets, API links on reports, settings tab state toggles, save buttons) MUST remain functional. Only styling/tokens and history data source are updated.
- **No backend changes:** Reports page uses `NEXT_PUBLIC_API_URL` for hrefs — these are left untouched.
- **Color token consistency:** After changes, every page will use the exact same hex codes as the home dashboard for backgrounds, borders, text, accents.

---

## Validation

1. **Build check** — `cd frontend && npm run build` completes without errors.
2. **TypeScript check** — No TS errors via IDE diagnostics or `npx tsc --noEmit`.
3. **Navigation check** (5 items):
   - Sidebar (md+): All 5 links navigate, correct active highlight.
   - Desktop Header: Shows correct page title for each route.
   - Mobile Bottom Nav: All 5 tabs navigate, correct active color.
   - Mobile Slide-over (MobileHeader): All 5 items work, drawer closes on click.
4. **Each page renders** without console errors:
   - `/history` — shows entries from localStorage or empty state.
   - `/datasets` — shows 6 dataset cards + upload section.
   - `/reports` — shows 3 report cards with View/Download buttons.
   - `/settings` — 4 tabs work, Profile/Application/Models/Security render correctly.
5. **Visual consistency** — All cards share the same bg `#161B22`, borders `rgba(48,54,61,0.8)`, border-radius 12px, text colors.

---

## Risks

| Risk | Handling |
|---|---|
| History page shows empty on fresh install (no localStorage entries) | Add explicit empty state card matching dashboard empty state pattern (icon + heading + subtext) |
| Importing types from `SessionHistory.tsx` causes circular references | If needed, inline the `HistoryEntry` interface / `STORAGE_KEY` string in history/page.tsx (identical copy) |
| Generic Tailwind classes are referenced elsewhere | Only update classes within the 4 target page files; globals.css `.card` alias and `elevated`/`primary` helpers keep working for other components |
| Build fails due to className typos after refactor | Run `GetDiagnostics` after edits; fix any Tailwind/TS issues before finalizing |
