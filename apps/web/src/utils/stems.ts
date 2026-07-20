import type { FocusSegment, HitPoint, StemDescriptor, StemKind } from '../types';
import { cssVar, type CssVariableName } from './designTokens';

export const STEM_ORDER: readonly StemKind[] = ['mix', 'vocals', 'drums', 'bass', 'other'];

export const STEM_LABELS: Record<StemKind, string> = {
  mix: '混音',
  vocals: '人声',
  drums: '鼓',
  bass: '贝斯',
  other: '其他 / 主旋律',
};

export const STEM_COLOR_TOKENS: Record<StemKind, CssVariableName> = {
  mix: '--stem-mix',
  vocals: '--stem-vocals',
  drums: '--stem-drums',
  bass: '--stem-bass',
  other: '--stem-other',
};

export const STEM_COLORS: Record<StemKind, string> = {
  mix: cssVar(STEM_COLOR_TOKENS.mix),
  vocals: cssVar(STEM_COLOR_TOKENS.vocals),
  drums: cssVar(STEM_COLOR_TOKENS.drums),
  bass: cssVar(STEM_COLOR_TOKENS.bass),
  other: cssVar(STEM_COLOR_TOKENS.other),
};

export interface StemLane {
  source: StemKind;
  top: number;
  bottom: number;
  center: number;
  height: number;
}

export interface StemTimelineLayout {
  height: number;
  rulerHeight: number;
  focusTop: number;
  focusBottom: number;
  lanesTop: number;
  lanes: StemLane[];
}

export function isStemKind(value: unknown): value is StemKind {
  return typeof value === 'string' && (STEM_ORDER as readonly string[]).includes(value);
}

export function primaryStemOf(point: Pick<HitPoint, 'primaryStem'> | Partial<HitPoint>): StemKind {
  return isStemKind(point.primaryStem) ? point.primaryStem : 'mix';
}

export function availableStemKinds(stems: StemDescriptor[] | null | undefined): StemKind[] {
  const available = new Set<StemKind>(['mix']);
  for (const stem of stems ?? []) {
    if (stem.available && isStemKind(stem.source)) available.add(stem.source);
  }
  return STEM_ORDER.filter((source) => available.has(source));
}

export function resolveVisibleStemKinds(
  stems: StemDescriptor[] | null | undefined,
  requested: readonly StemKind[] | null | undefined,
): StemKind[] {
  const available = new Set(availableStemKinds(stems));
  const requestedSet = new Set((requested ?? []).filter(isStemKind));
  const resolved = STEM_ORDER.filter((source) => available.has(source) && requestedSet.has(source));
  return resolved.length ? resolved : ['mix'];
}

export function buildStemLaneLayout(
  sources: readonly StemKind[],
  height = 394,
  rulerHeight = 34,
  focusHeight = 22,
): StemTimelineLayout {
  const safeSources = sources.length ? [...sources] : ['mix' as const];
  const focusTop = rulerHeight;
  const focusBottom = focusTop + focusHeight;
  const lanesTop = focusBottom;
  const laneHeight = Math.max(1, (height - lanesTop) / safeSources.length);
  const lanes = safeSources.map((source, index) => {
    const top = lanesTop + index * laneHeight;
    const bottom = index === safeSources.length - 1 ? height : lanesTop + (index + 1) * laneHeight;
    return { source, top, bottom, center: (top + bottom) / 2, height: bottom - top };
  });
  return { height, rulerHeight, focusTop, focusBottom, lanesTop, lanes };
}

export function timelineXForSample(
  sample: number,
  sampleRate: number,
  pixelsPerSecond: number,
  scrollLeft: number,
): number {
  return (sample / sampleRate) * pixelsPerSecond - scrollLeft;
}

export function focusSegmentAtSample(
  focusMap: readonly FocusSegment[] | null | undefined,
  sample: number,
): FocusSegment | undefined {
  return (focusMap ?? []).find(
    (segment) => segment.endSample > segment.startSample
      && sample >= segment.startSample
      && sample < segment.endSample,
  );
}
