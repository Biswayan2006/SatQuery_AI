"use client";

import { useState, useCallback } from "react";
import axios, { AxiosError } from "axios";
import toast from "react-hot-toast";
import type {
  AnalysisResponse,
  AnalysisState,
  ImageUploadResponse,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Upload ─────────────────────────────────────────────────────────────────

export function useImageUpload() {
  const [uploading, setUploading] = useState(false);

  const uploadImage = useCallback(
    async (file: File, modality: string = "auto"): Promise<ImageUploadResponse> => {
      setUploading(true);
      const formData = new FormData();
      formData.append("file", file);
      formData.append("modality", modality);

      try {
        const { data } = await axios.post<ImageUploadResponse>(
          `${API_BASE}/api/upload`,
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
            timeout: 30_000,
          }
        );
        return data;
      } catch (err) {
        const msg = extractErrorMessage(err);
        throw new Error(msg);
      } finally {
        setUploading(false);
      }
    },
    []
  );

  return { uploadImage, uploading };
}

// ── Analysis ───────────────────────────────────────────────────────────────

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    loading: false,
    result: null,
    error: null,
  });

  const analyze = useCallback(
    async (imageIds: string[], query: string, taskHint?: string) => {
      if (!imageIds.length || !query.trim()) {
        toast.error("Please upload an image and enter a query.");
        return;
      }

      setState({ loading: true, result: null, error: null });

      try {
        const { data } = await axios.post<AnalysisResponse>(
          `${API_BASE}/api/analyze`,
          {
            image_ids: imageIds,
            query: query.trim(),
            task_hint: taskHint,
          },
          { timeout: 120_000 }
        );
        setState({ loading: false, result: data, error: null });
        toast.success("Analysis complete");
        return data;
      } catch (err) {
        const msg = extractErrorMessage(err);
        setState({ loading: false, result: null, error: msg });
        toast.error(`Analysis failed: ${msg}`);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState({ loading: false, result: null, error: null });
  }, []);

  return { ...state, analyze, reset };
}

// ── Report download ────────────────────────────────────────────────────────

export function useReportDownload() {
  const [downloading, setDownloading] = useState(false);

  const downloadReport = useCallback(async (sessionId: string) => {
    if (!sessionId) return;
    setDownloading(true);

    try {
      const response = await axios.get(`${API_BASE}/api/report/${sessionId}`, {
        responseType: "blob",
        timeout: 30_000,
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.download = `satquery_report_${sessionId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      toast.success("Report downloaded");
    } catch (err) {
      toast.error("Failed to download report");
    } finally {
      setDownloading(false);
    }
  }, []);

  return { downloadReport, downloading };
}

// ── Health check ───────────────────────────────────────────────────────────

export async function checkHealth() {
  try {
    const { data } = await axios.get(`${API_BASE}/api/health`, { timeout: 5_000 });
    return data;
  } catch {
    return null;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (typeof detail === "object") return JSON.stringify(detail);
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "Unknown error";
}
