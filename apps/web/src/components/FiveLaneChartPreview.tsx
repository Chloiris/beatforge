import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChartDocument } from '../types';
import {
  chartApproachSeconds,
  DEFAULT_CHART_SCROLL_SPEED,
  chartNoteY,
  FIVE_PANEL_LABELS,
  FIVE_PANEL_LANES,
  flattenFivePanelEvents,
  visibleChartNotes,
  type ChartScrollSpeed,
  type FivePanelLane,
} from '../utils/chartPreview';
import { createCssColorResolver, cssVar, type CssVariableName } from '../utils/designTokens';

const CANVAS_HEIGHT = 590;
const CANVAS_WIDTH = 520;
const CHART_TOKENS = {
  background: '--chart-canvas-bg',
  lane: '--chart-lane',
  laneAlternate: '--chart-lane-alternate',
  divider: '--chart-lane-divider',
  receptor: '--chart-receptor',
  judgment: '--chart-judgment',
  hold: '--chart-hold',
  label: '--canvas-label',
  mine: '--danger',
} satisfies Record<string, CssVariableName>;

const LANE_COLOR_TOKENS: Record<FivePanelLane, CssVariableName> = {
  0: '--chart-note-left-down',
  1: '--chart-note-left-up',
  2: '--chart-note-center',
  3: '--chart-note-right-up',
  4: '--chart-note-right-down',
};

function resolveChartColors() {
  const resolve = createCssColorResolver();
  return {
    palette: Object.fromEntries(
      Object.entries(CHART_TOKENS).map(([key, token]) => [key, resolve(cssVar(token))]),
    ) as Record<keyof typeof CHART_TOKENS, string>,
    lanes: Object.fromEntries(
      FIVE_PANEL_LANES.map((lane) => [lane, resolve(cssVar(LANE_COLOR_TOKENS[lane]))]),
    ) as Record<FivePanelLane, string>,
  };
}

function diamondPath(context: CanvasRenderingContext2D, x: number, y: number, radius: number) {
  context.beginPath();
  context.moveTo(x, y - radius);
  context.lineTo(x + radius, y);
  context.lineTo(x, y + radius);
  context.lineTo(x - radius, y);
  context.closePath();
}

interface FiveLaneChartPreviewProps {
  chart: ChartDocument;
  currentTimeSec: number;
  scrollSpeed?: ChartScrollSpeed;
}

export function FiveLaneChartPreview({
  chart,
  currentTimeSec,
  scrollSpeed = DEFAULT_CHART_SCROLL_SPEED,
}: FiveLaneChartPreviewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const responsiveHeightRef = useRef(false);
  const [canvasSize, setCanvasSize] = useState({ width: CANVAS_WIDTH, height: CANVAS_HEIGHT });
  const colors = useMemo(resolveChartColors, []);
  const notes = useMemo(() => flattenFivePanelEvents(chart.events), [chart.events]);
  const approachSeconds = chartApproachSeconds(scrollSpeed);
  const visible = useMemo(
    () => visibleChartNotes(notes, currentTimeSec, approachSeconds),
    [approachSeconds, currentTimeSec, notes],
  );

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const update = () => {
      const measuredHeight = Math.round(wrap.clientHeight);
      const renderedCanvasHeight = Math.round(canvasRef.current?.clientHeight ?? 0);
      if (
        measuredHeight > 0
        && renderedCanvasHeight > 0
        && Math.abs(measuredHeight - renderedCanvasHeight) > 1
      ) {
        responsiveHeightRef.current = true;
      }
      const nextSize = {
        width: Math.max(280, Math.round(wrap.clientWidth || CANVAS_WIDTH)),
        height: responsiveHeightRef.current
          ? Math.max(160, measuredHeight)
          : CANVAS_HEIGHT,
      };
      setCanvasSize((currentSize) => (
        currentSize.width === nextSize.width && currentSize.height === nextSize.height
          ? currentSize
          : nextSize
      ));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = canvasSize;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const laneWidth = width / FIVE_PANEL_LANES.length;
    const judgmentLineY = Math.min(112, Math.max(72, height * 0.19));
    const travelDistance = Math.max(0, height - judgmentLineY - 34);
    context.fillStyle = colors.palette.background;
    context.fillRect(0, 0, width, height);

    FIVE_PANEL_LANES.forEach((lane) => {
      const left = lane * laneWidth;
      context.fillStyle = lane % 2 === 0 ? colors.palette.lane : colors.palette.laneAlternate;
      context.fillRect(left, 0, laneWidth, height);
      if (lane > 0) {
        context.strokeStyle = colors.palette.divider;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(Math.round(left) + 0.5, 0);
        context.lineTo(Math.round(left) + 0.5, height);
        context.stroke();
      }
      const centerX = left + laneWidth / 2;
      context.fillStyle = colors.palette.label;
      context.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(FIVE_PANEL_LABELS[lane], centerX, 28);
      context.strokeStyle = colors.palette.receptor;
      context.lineWidth = 2;
      diamondPath(context, centerX, judgmentLineY, Math.min(18, laneWidth * 0.2));
      context.stroke();
    });

    context.strokeStyle = colors.palette.judgment;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(0, judgmentLineY + 0.5);
    context.lineTo(width, judgmentLineY + 0.5);
    context.stroke();

    for (const note of visible) {
      const centerX = (note.lane + 0.5) * laneWidth;
      const startY = chartNoteY(
        note.startTimeSec,
        currentTimeSec,
        judgmentLineY,
        travelDistance,
        approachSeconds,
      );
      const radius = Math.max(9, Math.min(17, laneWidth * 0.18));
      if (note.type === 'hold' && note.endTimeSec !== null) {
        const endY = chartNoteY(
          note.endTimeSec,
          currentTimeSec,
          judgmentLineY,
          travelDistance,
          approachSeconds,
        );
        const top = Math.max(-radius, Math.min(startY, endY));
        const bottom = Math.min(height + radius, Math.max(startY, endY));
        context.globalAlpha = 0.74;
        context.fillStyle = colors.palette.hold;
        context.fillRect(centerX - radius * 0.34, top, radius * 0.68, Math.max(3, bottom - top));
        context.globalAlpha = 1;
      }
      if (startY < -radius * 2 || startY > height + radius * 2) continue;
      if (note.type === 'mine') {
        context.strokeStyle = colors.palette.mine;
        context.lineWidth = 3;
        context.beginPath();
        context.moveTo(centerX - radius, startY - radius);
        context.lineTo(centerX + radius, startY + radius);
        context.moveTo(centerX + radius, startY - radius);
        context.lineTo(centerX - radius, startY + radius);
        context.stroke();
        continue;
      }
      context.fillStyle = colors.lanes[note.lane];
      context.strokeStyle = colors.palette.receptor;
      context.lineWidth = note.pattern ? 2.5 : 1.25;
      diamondPath(context, centerX, startY, radius);
      context.fill();
      context.stroke();
      if (note.pattern) {
        context.fillStyle = colors.palette.background;
        context.beginPath();
        context.arc(centerX, startY, 3, 0, Math.PI * 2);
        context.fill();
      }
    }
  }, [approachSeconds, canvasSize, colors, currentTimeSec, visible]);

  return (
    <div
      className="five-lane-preview"
      ref={wrapRef}
      data-testid="five-lane-preview"
      data-current-time={currentTimeSec.toFixed(3)}
      data-scroll-speed={scrollSpeed}
      data-approach-seconds={approachSeconds.toFixed(3)}
    >
      <canvas
        ref={canvasRef}
        data-testid="five-lane-chart-canvas"
        aria-label={`五轨谱面预览：${chart.title}。音符从下向上移动，判定线位于上方。`}
      />
    </div>
  );
}
