"use client";

import { useState, useRef, useEffect } from "react";
import { Bell, CheckCircle, FileText, Database, AlertCircle, X, Check } from "lucide-react";

export interface NotificationItem {
  id: string;
  title: string;
  description: string;
  timestamp: string;
  read: boolean;
  type: "system" | "report" | "dataset" | "alert";
}

const INITIAL_NOTIFICATIONS: NotificationItem[] = [
  {
    id: "notif-1",
    title: "Model registry loaded",
    description: "5 specialist models active: BLIP-2, OWL-ViT, ResNet-50 ready for inference",
    timestamp: "8m ago",
    read: false,
    type: "system",
  },
  {
    id: "notif-2",
    title: "Report generated",
    description: "Urban_Watch_Change_2026-09-04 compiled with visual change heatmap",
    timestamp: "24m ago",
    read: false,
    type: "report",
  },
  {
    id: "notif-3",
    title: "Satellite pass imported",
    description: "12 new Sentinel-2 cloud-free tiles synchronized for Urban Watch",
    timestamp: "1h ago",
    read: true,
    type: "dataset",
  },
  {
    id: "notif-4",
    title: "Change detection alert",
    description: "Built-up surface area increased by 14.2% across target sector 4B",
    timestamp: "2h ago",
    read: true,
    type: "alert",
  },
];

export default function NotificationsPopover() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<NotificationItem[]>(INITIAL_NOTIFICATIONS);
  const [filter, setFilter] = useState<"all" | "unread">("all");
  const popoverRef = useRef<HTMLDivElement>(null);

  const unreadCount = items.filter((i) => !i.read).length;

  const markAllRead = () => {
    setItems((prev) => prev.map((item) => ({ ...item, read: true })));
  };

  const clearAll = () => {
    setItems([]);
  };

  const markItemRead = (id: string) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, read: true } : item))
    );
  };

  const removeItem = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setItems((prev) => prev.filter((item) => item.id !== id));
  };

  // Close on click outside or Escape
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };

    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const displayedItems =
    filter === "unread" ? items.filter((i) => !i.read) : items;

  const getTypeIcon = (type: NotificationItem["type"]) => {
    switch (type) {
      case "system":
        return <CheckCircle className="w-3.5 h-3.5" style={{ color: "var(--veg)" }} />;
      case "report":
        return <FileText className="w-3.5 h-3.5" style={{ color: "var(--accent)" }} />;
      case "dataset":
        return <Database className="w-3.5 h-3.5" style={{ color: "var(--water)" }} />;
      case "alert":
        return <AlertCircle className="w-3.5 h-3.5" style={{ color: "var(--change)" }} />;
    }
  };

  return (
    <div className="relative" ref={popoverRef}>
      {/* Bell Button */}
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="relative flex items-center justify-center w-10 h-10 rounded-lg
                   text-ink-muted hover:text-ink hover:bg-raised transition-colors no-tap"
        style={{ border: "1px solid var(--border)" }}
        aria-label="Notifications"
        aria-expanded={open}
      >
        <Bell className="w-[18px] h-[18px]" />
        {unreadCount > 0 && (
          <span
            className="absolute top-2 right-2 w-2 h-2 rounded-full"
            style={{ background: "var(--accent)", boxShadow: "0 0 6px var(--accent)" }}
          />
        )}
      </button>

      {/* Popover Card */}
      {open && (
        <div
          className="fixed sm:absolute inset-x-3 sm:inset-auto sm:right-0 top-16 sm:top-auto sm:mt-2 w-auto sm:w-96 rounded-xl shadow-2xl z-50 overflow-hidden flex flex-col animate-fade-in"
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            boxShadow: "0 12px 32px rgba(0, 0, 0, 0.45)",
          }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                Notifications
              </span>
              {unreadCount > 0 && (
                <span
                  className="px-2 py-0.5 text-[10px] font-bold rounded-full"
                  style={{
                    background: "color-mix(in srgb, var(--accent) 15%, transparent)",
                    color: "var(--accent)",
                    border: "1px solid color-mix(in srgb, var(--accent) 30%, transparent)",
                  }}
                >
                  {unreadCount} new
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {unreadCount > 0 && (
                <button
                  onClick={markAllRead}
                  className="text-[11px] font-medium px-2 py-1 rounded-lg hover:bg-raised transition-colors no-tap flex items-center gap-1"
                  style={{ color: "var(--text-muted)" }}
                  title="Mark all as read"
                >
                  <Check className="w-3 h-3" />
                  Mark all read
                </button>
              )}
              <button
                onClick={() => setOpen(false)}
                className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-raised transition-colors text-ink-muted no-tap"
                aria-label="Close notifications"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Filter Bar */}
          <div
            className="flex items-center px-4 py-1.5 gap-2"
            style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-surface)" }}
          >
            <button
              onClick={() => setFilter("all")}
              className={`text-xs px-2.5 py-1 rounded-lg font-medium transition-colors no-tap ${
                filter === "all" ? "text-accent bg-accent-soft" : "text-ink-muted hover:text-ink"
              }`}
            >
              All ({items.length})
            </button>
            <button
              onClick={() => setFilter("unread")}
              className={`text-xs px-2.5 py-1 rounded-lg font-medium transition-colors no-tap ${
                filter === "unread" ? "text-accent bg-accent-soft" : "text-ink-muted hover:text-ink"
              }`}
            >
              Unread ({unreadCount})
            </button>
          </div>

          {/* Notifications List */}
          <div className="max-h-[320px] overflow-y-auto divide-y divide-line">
            {displayedItems.length === 0 ? (
              <div className="py-8 text-center px-4">
                <Bell className="w-6 h-6 mx-auto mb-2 opacity-30" style={{ color: "var(--text-muted)" }} />
                <p className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
                  No notifications to display
                </p>
              </div>
            ) : (
              displayedItems.map((item) => (
                <div
                  key={item.id}
                  onClick={() => markItemRead(item.id)}
                  className={`flex items-start gap-3 p-3 text-left transition-colors cursor-pointer hover:bg-raised group ${
                    !item.read ? "bg-[color-mix(in_srgb,var(--accent)_4%,transparent)]" : ""
                  }`}
                >
                  <div className="mt-0.5 flex-shrink-0">{getTypeIcon(item.type)}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1 mb-0.5">
                      <p
                        className={`text-xs font-medium truncate ${
                          !item.read ? "font-semibold" : ""
                        }`}
                        style={{ color: "var(--text-primary)" }}
                      >
                        {item.title}
                      </p>
                      <span className="text-[10px] whitespace-nowrap" style={{ color: "var(--text-faint)" }}>
                        {item.timestamp}
                      </span>
                    </div>
                    <p className="text-[11px] leading-relaxed line-clamp-2" style={{ color: "var(--text-muted)" }}>
                      {item.description}
                    </p>
                  </div>
                  <button
                    onClick={(e) => removeItem(item.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-inset transition-opacity text-ink-faint hover:text-ink-muted"
                    aria-label="Dismiss notification"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Footer */}
          {items.length > 0 && (
            <div
              className="flex items-center justify-between px-4 py-2"
              style={{ borderTop: "1px solid var(--border)", background: "var(--bg-surface)" }}
            >
              <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>
                Real-time satellite updates
              </span>
              <button
                onClick={clearAll}
                className="text-xs font-medium text-ink-muted hover:text-danger transition-colors no-tap"
              >
                Clear all
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
