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
  onAddImage: (file: File) => void;
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

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.slice(0, maxImages - images.length).forEach((file) => {
        onAddImage(file);
      });
    },
    [images.length, maxImages, onAddImage]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/tiff": [".tif", ".tiff"],
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
    },
    disabled: !canAddMore,
    multiple: true,
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
          {...getRootProps()}
          className={`
            relative cursor-pointer rounded-xl border-2 border-dashed
            transition-all duration-300 p-6
            ${isDragActive
              ? "border-satellite-500 bg-satellite-500/5 dropzone-active"
              : "border-slate-600/50 hover:border-satellite-600/50 hover:bg-slate-800/30"
            }
            ${!canAddMore ? "opacity-40 cursor-not-allowed" : ""}
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center border border-slate-600/50">
              <Upload
                className={`w-5 h-5 transition-colors ${
                  isDragActive ? "text-satellite-400" : "text-slate-400"
                }`}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-300">
                {isDragActive
                  ? "Drop your image here"
                  : images.length === 0
                  ? "Upload satellite image"
                  : "Add second image (for change detection / fusion)"}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                GeoTIFF, TIFF, PNG, JPEG — up to 50 MB
              </p>
            </div>
          </div>

          {/* Scan line on drag */}
          {isDragActive && (
            <div className="absolute inset-0 overflow-hidden rounded-xl pointer-events-none">
              <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-satellite-400/60 to-transparent animate-scan-line" />
            </div>
          )}
        </motion.div>
      )}

      {/* Format hint */}
      <div className="flex flex-wrap gap-1.5">
        {[".tif", ".tiff", ".png", ".jpg"].map((ext) => (
          <span
            key={ext}
            className="px-2 py-0.5 rounded text-xs bg-slate-800/60 text-slate-500 border border-slate-700/40 font-mono"
          >
            {ext}
          </span>
        ))}
        <span className="px-2 py-0.5 rounded text-xs bg-slate-800/60 text-slate-500 border border-slate-700/40">
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
      className="glass-card overflow-hidden"
    >
      <div className="flex gap-3 p-3">
        {/* Preview */}
        <div className="relative w-20 h-20 flex-shrink-0 rounded-lg overflow-hidden bg-slate-800 border border-slate-700/50">
          {image.previewUrl ? (
            <img
              src={image.previewUrl}
              alt={image.file.name}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <ImageIcon className="w-8 h-8 text-slate-500" />
            </div>
          )}
          <div className="absolute top-1 left-1 bg-space-900/80 text-satellite-400 text-xs font-bold px-1.5 rounded">
            {index}
          </div>
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm font-medium text-slate-200 truncate">{image.file.name}</p>
            <button
              onClick={onRemove}
              className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-700/60 hover:bg-red-900/50 flex items-center justify-center transition-colors"
            >
              <X className="w-3.5 h-3.5 text-slate-400 hover:text-red-400" />
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
            {image.uploading ? (
              <div className="flex items-center gap-1.5 text-xs text-satellite-400">
                <div className="flex gap-0.5">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="w-1 h-1 rounded-full bg-satellite-400 loading-dot"
                      style={{ animationDelay: `${i * 0.2}s` }}
                    />
                  ))}
                </div>
                <span>Uploading…</span>
              </div>
            ) : image.error ? (
              <span className="text-xs text-red-400">{image.error}</span>
            ) : resp ? (
              <>
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${modalityConfig.className}`}
                >
                  {modalityConfig.icon}
                  {modalityConfig.label}
                </span>

                {resp.shape?.length >= 2 && (
                  <span className="text-xs text-slate-500 font-mono">
                    {resp.shape[1]}×{resp.shape[0]}
                  </span>
                )}

                {resp.bands > 0 && (
                  <span className="text-xs text-slate-500 font-mono">
                    {resp.bands} band{resp.bands !== 1 ? "s" : ""}
                  </span>
                )}

                {resp.is_geotiff && (
                  <span className="px-1.5 py-0.5 rounded text-xs bg-emerald-900/40 text-emerald-400 border border-emerald-700/30">
                    GeoTIFF
                  </span>
                )}

                <span className="text-xs text-slate-600">
                  {image.file_size_kb
                    ? `${(image.file.size / 1024).toFixed(0)} KB`
                    : `${(image.file.size / 1024).toFixed(0)} KB`}
                </span>
              </>
            ) : null}
          </div>

          {resp?.crs && (
            <p className="text-xs text-slate-600 mt-1 truncate font-mono">
              CRS: {resp.crs}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}
