"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, ImageIcon, Satellite, Layers, HelpCircle } from "lucide-react";
import type { UploadedImage } from "@/types";

const MODALITY_CONFIG = {
  optical: {
    label: "Optical",
    className: "badge-optical",
    icon: <ImageIcon className="w-3 h-3" />,
  },
  sar: {
    label: "SAR",
    className: "badge-sar",
    icon: <Satellite className="w-3 h-3" />,
  },
  multispectral: {
    label: "Multispectral",
    className: "badge-multispectral",
    icon: <Layers className="w-3 h-3" />,
  },
  unknown: {
    label: "Unknown",
    className: "badge-unknown",
    icon: <HelpCircle className="w-3 h-3" />,
  },
} as const;

interface Props {
  images: UploadedImage[];
  onAddImage: (file: File, modality: "auto" | "optical" | "sar" | "multispectral") => void;
  onRemoveImage: (index: number) => void;
  maxImages?: number;
}

export default function ImageUpload({
  images,
  onAddImage,
  onRemoveImage,
  maxImages = 2,
}: Props) {
  const canAddMore = images.length < maxImages;
  const [selectedModality, setSelectedModality] = useState<"auto" | "optical" | "sar" | "multispectral">("auto");

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.slice(0, maxImages - images.length).forEach((file) => {
        onAddImage(file, selectedModality);
      });
    },
    [images.length, maxImages, onAddImage, selectedModality]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      "image/tiff": [".tif", ".tiff"],
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
    },
    disabled: !canAddMore,
    multiple: true,
    noClick: true,
  });

  return (
    <div className="flex flex-col gap-4">
      {/* Uploaded images grid */}
      <AnimatePresence>
        {images.map((img, idx) => (
          <ImageCard
            key={`${img.file.name}-${idx}`}
            image={img}
            index={idx + 1}
            onRemove={() => onRemoveImage(idx)}
          />
        ))}
      </AnimatePresence>

      {/* Dropzone */}
      {canAddMore && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className={`relative cursor-pointer rounded-xl border-2 border-dashed
            transition-all duration-300 p-6 ${isDragActive ? "dropzone-active" : ""}
            ${!canAddMore ? "opacity-40 cursor-not-allowed" : ""}
          `}
          style={{
            borderColor: isDragActive ? "var(--accent)" : "var(--border-strong)",
            background: isDragActive ? "var(--accent-soft)" : "transparent",
          }}
          {...(getRootProps() as Record<string, unknown>)}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-3 text-center">
            <label className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
              Image type
              <select
                value={selectedModality}
                onChange={(event) => setSelectedModality(event.target.value as typeof selectedModality)}
                onClick={(event) => event.stopPropagation()}
                className="rounded-md px-2 py-1 text-xs"
                style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
              >
                <option value="auto">Auto-detect</option>
                <option value="optical">Optical</option>
                <option value="sar">SAR / radar</option>
                <option value="multispectral">Multispectral</option>
              </select>
            </label>
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center"
              style={{ background: "var(--bg-inset)", border: "1px solid var(--border)" }}
            >
              <Upload
                className="w-5 h-5 transition-colors"
                style={{ color: isDragActive ? "var(--accent)" : "var(--text-muted)" }}
              />
            </div>
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                {isDragActive
                  ? "Drop your image here"
                  : images.length === 0
                  ? "Upload satellite image"
                  : "Add second image (for change detection / fusion)"}
              </p>
              <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                GeoTIFF, TIFF, PNG, JPEG — up to 50 MB
              </p>
            </div>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                open();
              }}
              className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition-colors"
              style={{
                background: "var(--accent)",
                color: "var(--accent-contrast)",
              }}
            >
              <Upload className="w-3.5 h-3.5" />
              Choose images
            </button>
          </div>

          {/* Scan line on drag */}
          {isDragActive && (
            <div className="absolute inset-0 overflow-hidden rounded-xl pointer-events-none">
              <div
                className="absolute inset-x-0 h-0.5 animate-scan-line"
                style={{ background: "linear-gradient(to right, transparent, color-mix(in srgb, var(--accent) 65%, transparent), transparent)" }}
              />
            </div>
          )}
        </motion.div>
      )}

      {/* Format hint */}
      <div className="flex flex-wrap gap-1.5">
        {[".tif", ".tiff", ".png", ".jpg"].map((ext) => (
          <span
            key={ext}
            className="px-2 py-0.5 rounded text-xs font-mono"
            style={{ background: "var(--bg-raised)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
          >
            {ext}
          </span>
        ))}
        <span
          className="px-2 py-0.5 rounded text-xs"
          style={{ background: "var(--bg-raised)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
        >
          GeoTIFF supported
        </span>
      </div>
    </div>
  );
}

// ── Image Card ────────────────────────────────────────────────────────────────

function ImageCard({
  image,
  index,
  onRemove,
}: {
  image: UploadedImage;
  index: number;
  onRemove: () => void;
}) {
  const resp = image.uploadResponse;
  const modality = resp?.modality ?? "unknown";
  const modalityConfig = MODALITY_CONFIG[modality] ?? MODALITY_CONFIG.unknown;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className="sq-card overflow-hidden"
    >
      <div className="flex gap-3 p-3">
        {/* Preview */}
        <div
          className="relative w-20 h-20 flex-shrink-0 rounded-lg overflow-hidden"
          style={{ background: "var(--bg-inset)", border: "1px solid var(--border)" }}
        >
          {image.previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={image.previewUrl}
              alt={image.file.name}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <ImageIcon className="w-8 h-8" style={{ color: "var(--text-faint)" }} />
            </div>
          )}
          <div
            className="absolute top-1 left-1 text-xs font-bold px-1.5 rounded"
            style={{ background: "color-mix(in srgb, var(--bg-base) 82%, transparent)", color: "var(--accent)" }}
          >
            {index}
          </div>
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>{image.file.name}</p>
            <button
              onClick={onRemove}
              className="group flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center transition-colors
                         hover:bg-[color-mix(in_srgb,var(--danger)_18%,transparent)]"
              style={{ background: "var(--bg-raised)" }}
              aria-label="Remove image"
            >
              <X className="w-3.5 h-3.5 transition-colors group-hover:text-[var(--danger)]" style={{ color: "var(--text-muted)" }} />
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
            {image.uploading ? (
              <div className="flex items-center gap-1.5 text-xs" style={{ color: "var(--accent)" }}>
                <div className="flex gap-0.5">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="w-1 h-1 rounded-full loading-dot"
                      style={{ background: "var(--accent)", animationDelay: `${i * 0.2}s` }}
                    />
                  ))}
                </div>
                <span>Uploading…</span>
              </div>
            ) : image.error ? (
              <span className="text-xs" style={{ color: "var(--danger)" }}>{image.error}</span>
            ) : resp ? (
              <>
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${modalityConfig.className}`}
                >
                  {modalityConfig.icon}
                  {modalityConfig.label}
                </span>

                {resp.shape?.length >= 2 && (
                  <span className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                    {resp.shape[1]}×{resp.shape[0]}
                  </span>
                )}

                {resp.bands > 0 && (
                  <span className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                    {resp.bands} band{resp.bands !== 1 ? "s" : ""}
                  </span>
                )}

                {resp.is_geotiff && (
                  <span
                    className="px-1.5 py-0.5 rounded text-xs"
                    style={{ background: "color-mix(in srgb, var(--veg) 14%, transparent)", color: "var(--veg)", border: "1px solid color-mix(in srgb, var(--veg) 30%, transparent)" }}
                  >
                    GeoTIFF
                  </span>
                )}

                <span className="text-xs" style={{ color: "var(--text-faint)" }}>
                  {(image.file.size / 1024).toFixed(0)} KB
                </span>
              </>
            ) : null}
          </div>

          {resp?.crs && (
            <p className="text-xs mt-1 truncate font-mono" style={{ color: "var(--text-faint)" }}>
              CRS: {resp.crs}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}
