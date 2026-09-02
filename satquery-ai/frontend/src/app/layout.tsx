import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Toaster } from "react-hot-toast";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "SatQuery AI — Remote Sensing Intelligence",
  description:
    "Agentic vision-language assistant for multimodal satellite image analysis. VQA, captioning, change detection, and SAR-optical fusion.",
  keywords: [
    "satellite imagery",
    "remote sensing",
    "AI analysis",
    "change detection",
    "SAR",
    "multispectral",
    "VQA",
  ],
  authors: [{ name: "SatQuery AI" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#050811",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-space-950 text-slate-100 antialiased">
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: "#0f1a30",
              color: "#e2e8f0",
              border: "1px solid rgba(58,171,255,0.2)",
              borderRadius: "8px",
              fontSize: "14px",
            },
            success: {
              iconTheme: { primary: "#22c55e", secondary: "#0f1a30" },
            },
            error: {
              iconTheme: { primary: "#ef4444", secondary: "#0f1a30" },
            },
          }}
        />
        {children}
      </body>
    </html>
  );
}
