"use client";

import { useEffect } from "react";

type Shortcut = {
  key: string;
  meta?: boolean;
  ctrl?: boolean;
  shift?: boolean;
  handler: () => void;
};

export function useKeyboard(shortcuts: Shortcut[]) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        const metaMatch  = s.meta  ? (e.metaKey  || e.ctrlKey) : !s.meta;
        const ctrlMatch  = s.ctrl  ? e.ctrlKey  : true;
        const shiftMatch = s.shift ? e.shiftKey : true;
        if (e.key === s.key && metaMatch && ctrlMatch && shiftMatch) {
          e.preventDefault();
          s.handler();
          return;
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [shortcuts]);
}
