"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Search, Database, RefreshCw } from "lucide-react";

const DATASETS = [
  { name: "BigEarthNet-S2", type: "Multispectral", images: 590326, status: "synced"    },
  { name: "BigEarthNet-S1", type: "SAR",           images: 590326, status: "synced"    },
  { name: "Urban Watch",    type: "Optical",       images: 1240,   status: "synced"    },
  { name: "Coastal Survey", type: "SAR+Optical",   images: 340,    status: "uploading" },
  { name: "Forest 2024",    type: "Multispectral", images: 890,    status: "synced"    },
  { name: "Change Masks",   type: "Reference",     images: 4500,   status: "synced"    },
];

const TYPE_HUE: Record<string, string> = {
  Multispectral: "var(--accent)",
  SAR:           "var(--sar)",
  Optical:       "var(--water)",
  "SAR+Optical": "var(--bare)",
  Reference:     "var(--veg)",
};

export default function DatasetsPage() {
  const [search, setSearch] = useState("");

  return (
    <div className="max-w-[1400px] mx-auto space-y-5 animate-fade-in">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-lg font-bold text-ink">Datasets</h1>
          <p className="text-xs text-ink-muted mt-0.5">Manage training data and input imagery</p>
        </div>
      </div>

      {/* Add new data */}
      <div className="card p-5">
        <h2 className="text-sm font-bold text-ink mb-4">Add New Data</h2>
        <label className="block border-2 border-dashed border-line-strong rounded-xl p-8 text-center cursor-pointer hover:border-accent-border hover:bg-accent-soft transition-all mb-4 group">
          <input type="file" className="hidden" accept=".tif,.tiff" multiple />
          <Upload className="w-8 h-8 text-ink-faint group-hover:text-accent mx-auto mb-3 transition-colors" />
          <p className="text-sm font-semibold text-ink-soft group-hover:text-ink mb-1">
            Drop GeoTIFF / TIFF files here
          </p>
          <p className="text-xs text-ink-faint">
            or click to browse: supports Sentinel-1, Sentinel-2, Landsat, Cartosat
          </p>
        </label>
        <div className="flex gap-3 flex-wrap">
          {["Upload Optical", "Upload SAR", "Upload Reference Masks"].map(label => (
            <label key={label}>
              <input type="file" className="hidden" accept=".tif,.tiff" />
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-raised border border-line text-xs font-semibold text-ink-soft hover:text-ink hover:bg-inset cursor-pointer transition-all no-tap">
                <Upload className="w-3.5 h-3.5" />{label}
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Browse */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
          <h2 className="text-sm font-bold text-ink">Browse Datasets</h2>
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-raised border border-line w-64">
            <Search className="w-3.5 h-3.5 text-ink-muted flex-shrink-0" />
            <input
              placeholder="Search datasets…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="bg-transparent text-xs text-ink-soft placeholder:text-ink-faint outline-none w-full"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {DATASETS.filter(d =>
            d.name.toLowerCase().includes(search.toLowerCase())
          ).map((ds, i) => {
            const hue = TYPE_HUE[ds.type];
            return (
              <motion.div
                key={ds.name}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
                className="card-elevated p-4 hover:border-line-strong transition-all cursor-pointer group"
              >
                <div className="h-24 rounded-xl bg-inset border border-line flex items-center justify-center mb-3">
                  <Database className="w-8 h-8 text-ink-faint group-hover:text-ink-muted transition-colors" />
                </div>
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <p className="text-xs font-bold text-ink">{ds.name}</p>
                  <span
                    className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border flex-shrink-0 ${
                      hue ? "" : "text-ink-muted bg-raised border-line"
                    }`}
                    style={
                      hue
                        ? {
                            color: hue,
                            background: `color-mix(in srgb, ${hue} 13%, transparent)`,
                            borderColor: `color-mix(in srgb, ${hue} 32%, transparent)`,
                          }
                        : undefined
                    }
                  >
                    {ds.type}
                  </span>
                </div>
                <p className="text-[10px] text-ink-muted mb-2">
                  {ds.images.toLocaleString()} images
                </p>
                <div className="flex items-center gap-1.5">
                  {ds.status === "synced" ? (
                    <>
                      <div className="w-1.5 h-1.5 rounded-full bg-veg" />
                      <span className="text-[10px] text-veg">Synced</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-3 h-3 text-warning animate-spin" />
                      <span className="text-[10px] text-warning">Uploading…</span>
                    </>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
