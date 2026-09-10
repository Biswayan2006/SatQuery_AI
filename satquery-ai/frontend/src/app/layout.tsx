import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import { Toaster } from "react-hot-toast";
import AppShell from "@/components/layout/AppShell";
import { ThemeProvider } from "@/components/theme/ThemeProvider";
import { ProjectProvider } from "@/context/ProjectContext";
import AuthProvider from "@/components/AuthProvider";
import "./globals.css";

/* Type system: IBM Plex Sans for UI, IBM Plex Mono for data values only. */
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-plex-sans",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
  variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "SatQuery AI | PS 26167: ISRO / Space Applications Centre",
  description:
    "Agentic vision-language remote sensing assistant for optical and SAR satellite data. Problem Statement 26167 for Smart India Hackathon.",
  keywords: ["satellite imagery", "remote sensing", "AI analysis", "change detection", "SAR", "ISRO"],
  authors: [{ name: "SatQuery AI" }],
  manifest: "/manifest.json",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#F0F1EC",
};

/* Runs before first paint: applies the persisted (or system) theme so there is
   no flash of the wrong palette. Kept inline + minified on purpose. */
const themeScript = `(function(){try{var s=localStorage.getItem('sq-theme');var m=window.matchMedia('(prefers-color-scheme: dark)').matches;var t=(s==='light'||s==='dark')?s:(m?'dark':'light');var r=document.documentElement;r.setAttribute('data-theme',t);r.style.colorScheme=t;}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <body className="antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <AuthProvider>
          <ThemeProvider>
            <ProjectProvider>
              <Toaster
                position="top-right"
                toastOptions={{
                  style: {
                    background: "var(--bg-elevated)",
                    color: "var(--text-primary)",
                    border: "1px solid var(--border)",
                    borderRadius: "12px",
                    fontSize: "13px",
                    boxShadow: "var(--shadow-lg)",
                  },
                  success: { iconTheme: { primary: "var(--veg)", secondary: "var(--bg-elevated)" } },
                  error: { iconTheme: { primary: "var(--danger)", secondary: "var(--bg-elevated)" } },
                }}
              />
              <AppShell>{children}</AppShell>
            </ProjectProvider>
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
