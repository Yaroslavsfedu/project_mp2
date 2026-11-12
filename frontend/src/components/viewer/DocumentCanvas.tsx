import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Maximize2,
  RotateCw,
  ScanText,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useAuthedImage } from "@/api/useAuthedImage";
import { cn } from "@/lib/utils";
import type { BoundingBox, OcrPage } from "@/types";
import { Skeleton } from "@/components/ui";

interface Props {
  page: OcrPage;
  pageCount: number;
  pageNumber: number;
  onPageChange: (page: number) => void;
  selectedBoxId: string | null;
  onSelectBox: (box: BoundingBox) => void;
}

/** Цвет рамки по уверенности распознавания (stroke для SVG-оверлея). */
function bboxStroke(confidence: number): string {
  if (confidence >= 0.9) return "stroke-emerald-400";
  if (confidence >= 0.7) return "stroke-amber-400";
  return "stroke-rose-400";
}

function useContainerWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    setWidth(element.clientWidth);
    return () => observer.disconnect();
  }, []);

  return { ref, width };
}

export function DocumentCanvas({
  page,
  pageCount,
  pageNumber,
  onPageChange,
  selectedBoxId,
  onSelectBox,
}: Props) {
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [showBoxes, setShowBoxes] = useState(true);
  const [lowOnly, setLowOnly] = useState(false);
  const { ref, width } = useContainerWidth<HTMLDivElement>();
  const { url, loading, error } = useAuthedImage(page.image_url);

  useEffect(() => {
    setZoom(100);
    setRotation(0);
  }, [page.page_number]);

  const naturalWidth = page.image_width || 1000;
  const naturalHeight = page.image_height || 1400;
  const rotated = rotation === 90 || rotation === 270;

  const baseWidth = Math.max(width - 32, 200) * (zoom / 100);
  const baseHeight = (baseWidth * naturalHeight) / naturalWidth;
  const boxWidth = rotated ? baseHeight : baseWidth;
  const boxHeight = rotated ? baseWidth : baseHeight;

  const visibleBoxes = (page.bounding_boxes ?? []).filter(
    (box) => !lowOnly || box.confidence < 0.85,
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-1.5 border-b border-white/[0.06] px-3 py-2">
        <div className="flex items-center gap-1 rounded-lg border border-white/10 bg-ink-900/70 p-0.5">
          <button
            className="rounded p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
            onClick={() => onPageChange(pageNumber - 1)}
            disabled={pageNumber <= 1}
            title="Предыдущая страница"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="min-w-[64px] text-center text-xs text-slate-300">
            {pageNumber} / {pageCount || 1}
          </span>
          <button
            className="rounded p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
            onClick={() => onPageChange(pageNumber + 1)}
            disabled={pageNumber >= pageCount}
            title="Следующая страница"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-1 rounded-lg border border-white/10 bg-ink-900/70 p-0.5">
          <button
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
            onClick={() => setZoom((value) => Math.max(50, value - 25))}
            title="Уменьшить"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            className="rounded px-1 text-[11px] text-slate-400 hover:text-white"
            onClick={() => setZoom(100)}
            title="Сбросить масштаб"
          >
            {zoom}%
          </button>
          <button
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
            onClick={() => setZoom((value) => Math.min(300, value + 25))}
            title="Увеличить"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
            onClick={() => setZoom(100)}
            title="По ширине"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
          <button
            className="rounded p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
            onClick={() => setRotation((value) => (value + 90) % 360)}
            title="Повернуть страницу"
          >
            <RotateCw className="h-4 w-4" />
          </button>
        </div>

        <button
          onClick={() => setShowBoxes((value) => !value)}
          className={cn(
            "flex items-center gap-1.5 rounded-lg border px-2 py-1.5 text-[11px] transition",
            showBoxes
              ? "border-brand-500/40 bg-brand-500/15 text-brand-100"
              : "border-white/10 bg-ink-900/70 text-slate-400 hover:text-white",
          )}
          title="Показать/скрыть рамки распознанного текста"
        >
          {showBoxes ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
          Рамки
        </button>

        <button
          onClick={() => setLowOnly((value) => !value)}
          className={cn(
            "flex items-center gap-1.5 rounded-lg border px-2 py-1.5 text-[11px] transition",
            lowOnly
              ? "border-amber-500/40 bg-amber-500/15 text-amber-200"
              : "border-white/10 bg-ink-900/70 text-slate-400 hover:text-white",
          )}
          title="Показать только строки с низкой уверенностью"
        >
          <ScanText className="h-3.5 w-3.5" />
          &lt; 85%
        </button>

        <div className="ml-auto flex items-center gap-2 text-[11px] text-slate-400">
          <span className="rounded border border-white/10 bg-white/[0.03] px-1.5 py-0.5">
            {page.source === "native" ? "текстовый слой" : page.source === "needs_ocr" ? "нужен OCR" : "OCR"}
          </span>
          <span title="Средняя уверенность распознавания">
            conf {(page.avg_confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      <div ref={ref} className="relative flex-1 overflow-auto bg-[#0b0910] p-4">
        {loading && <Skeleton className="mx-auto h-[60vh] w-[70%]" />}
        {error && (
          <div className="mx-auto max-w-md rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
            {error}
          </div>
        )}
        {!loading && !error && url && (
          <div className="mx-auto" style={{ width: boxWidth, height: boxHeight }}>
            <div
              className="relative"
              style={{
                width: baseWidth,
                height: baseHeight,
                transform: `rotate(${rotation}deg)`,
                position: "absolute",
                left: "50%",
                top: "50%",
                marginLeft: -baseWidth / 2,
                marginTop: -baseHeight / 2,
              }}
            >
              <img
                src={url}
                alt={`Страница ${page.page_number}`}
                className="h-full w-full select-none rounded-sm bg-white shadow-2xl"
                draggable={false}
              />
              {showBoxes && visibleBoxes.length > 0 && (
                <svg
                  className="absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
                  preserveAspectRatio="none"
                >
                  {visibleBoxes.map((box) => {
                    const [x1, y1, x2, y2] = box.bbox;
                    const active = selectedBoxId === box.id;
                    return (
                      <g key={box.id} onClick={() => onSelectBox(box)} className="cursor-pointer">
                        <rect
                          x={x1}
                          y={y1}
                          width={Math.max(x2 - x1, 2)}
                          height={Math.max(y2 - y1, 2)}
                          rx={3}
                          className={cn(
                            "transition-all stroke-[1.5] hover:fill-brand-400/25",
                            active ? "stroke-brand-300 fill-brand-400" : bboxStroke(box.confidence),
                          )}
                          fillOpacity={active ? 0.3 : 0.12}
                        >
                          <title>{`${box.text}\nуверенность: ${(box.confidence * 100).toFixed(0)}%`}</title>
                        </rect>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
