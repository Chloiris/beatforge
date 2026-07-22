import type { ChartEvent, ChartNoteType } from '../types';

export const FIVE_PANEL_LANES = [0, 1, 2, 3, 4] as const;
export type FivePanelLane = (typeof FIVE_PANEL_LANES)[number];

export const CHART_SCROLL_SPEED_OPTIONS = [1, 2, 3, 4, 5, 6, 8] as const;
export type ChartScrollSpeed = (typeof CHART_SCROLL_SPEED_OPTIONS)[number];
export const DEFAULT_CHART_SCROLL_SPEED: ChartScrollSpeed = 4;
export const BASE_CHART_APPROACH_SECONDS = 2.6;

export const FIVE_PANEL_LABELS: Record<FivePanelLane, string> = {
  0: '左下',
  1: '左上',
  2: '中心',
  3: '右上',
  4: '右下',
};

export function isChartScrollSpeed(value: number): value is ChartScrollSpeed {
  return CHART_SCROLL_SPEED_OPTIONS.includes(value as ChartScrollSpeed);
}

export function chartApproachSeconds(
  scrollSpeed: number,
  baseApproachSeconds = BASE_CHART_APPROACH_SECONDS,
): number {
  const safeBase = Number.isFinite(baseApproachSeconds) && baseApproachSeconds > 0
    ? baseApproachSeconds
    : BASE_CHART_APPROACH_SECONDS;
  if (!Number.isFinite(scrollSpeed) || scrollSpeed <= 0) return safeBase;
  return safeBase / scrollSpeed;
}

export interface RenderableChartNote {
  key: string;
  lane: FivePanelLane;
  type: ChartNoteType;
  startTimeSec: number;
  endTimeSec: number | null;
  beat: number;
  measure: number;
  pattern: string | null;
}

export function flattenFivePanelEvents(events: readonly ChartEvent[]): RenderableChartNote[] {
  const notes: RenderableChartNote[] = [];
  events.forEach((event, eventIndex) => {
    event.notes.forEach((note, noteIndex) => {
      if (!FIVE_PANEL_LANES.includes(note.lane as FivePanelLane)) return;
      notes.push({
        key: `${event.sourceEventId ?? eventIndex}:${noteIndex}:${note.lane}`,
        lane: note.lane as FivePanelLane,
        type: note.type,
        startTimeSec: event.timeSec,
        endTimeSec: note.endTimeSec,
        beat: event.beat,
        measure: event.measure,
        pattern: event.pattern,
      });
    });
  });
  return notes.sort((left, right) => (
    left.startTimeSec - right.startTimeSec || left.lane - right.lane
  ));
}

export function visibleChartNotes(
  notes: readonly RenderableChartNote[],
  currentTimeSec: number,
  approachSeconds: number,
  missWindowSeconds = 0.35,
): RenderableChartNote[] {
  const latestStart = currentTimeSec + Math.max(0, approachSeconds);
  const earliestVisibleEnd = currentTimeSec - Math.max(0, missWindowSeconds);
  return notes.filter((note) => (
    note.startTimeSec <= latestStart
    && (note.endTimeSec ?? note.startTimeSec) >= earliestVisibleEnd
  ));
}

export function chartNoteY(
  eventTimeSec: number,
  currentTimeSec: number,
  judgmentLineY: number,
  travelDistance: number,
  approachSeconds: number,
): number {
  if (!Number.isFinite(approachSeconds) || approachSeconds <= 0) return judgmentLineY;
  return judgmentLineY
    + ((eventTimeSec - currentTimeSec) / approachSeconds) * Math.max(0, travelDistance);
}

export function clampPlaybackTime(timeSec: number, durationSec: number): number {
  if (!Number.isFinite(timeSec)) return 0;
  const safeDuration = Number.isFinite(durationSec) ? Math.max(0, durationSec) : 0;
  return Math.max(0, Math.min(safeDuration, timeSec));
}
