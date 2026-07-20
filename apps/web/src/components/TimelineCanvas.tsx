import { useEffect, useMemo, useRef, useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import { api } from '../api/client';
import { filterHitPoints, useEditorStore } from '../state/editorStore';
import type {
  AlignmentHierarchyUnit,
  AlignmentLayer,
  AlignmentMethodId,
  AlignmentToken,
  CandidateEvent,
  HitPoint,
  StemKind,
  TrackDetail,
  WaveformPeaks,
} from '../types';
import {
  calculateGridLines,
  gridSubdivisionCount,
  gridVisibilityForScale,
  nearestGridSample,
} from '../utils/grid';
import {
  buildStemLaneLayout,
  focusSegmentAtSample,
  primaryStemOf,
  resolveVisibleStemKinds,
  STEM_COLOR_TOKENS,
  STEM_LABELS,
  STEM_ORDER,
  timelineXForSample,
} from '../utils/stems';
import { formatTime, sampleToSeconds } from '../utils/time';
import { createCssColorResolver, cssVar, type CssVariableName } from '../utils/designTokens';

const EDIT_TIMELINE_HEIGHT = 394;
const ALIGNMENT_WAVEFORM_HEIGHT = 196;
const RULER_HEIGHT = 34;
const ALIGNMENT_LANE_HEIGHT = 38;
const BAND_COLORS: Record<HitPoint['band'], string> = {
  low_hit: cssVar('--band-low'),
  mid_hit: cssVar('--band-mid'),
  high_hit: cssVar('--band-high'),
  full_band_accent: cssVar('--band-accent'),
};
const CANDIDATE_COLORS: Record<CandidateEvent['status'], string> = {
  accepted: cssVar('--success'),
  uncertain: cssVar('--warning'),
  rejected: cssVar('--marker-muted'),
};
const EMPTY_ALIGNMENT_LANES: AlignmentTimelineLane[] = [];

export type AlignmentTimelineToken = AlignmentToken | AlignmentHierarchyUnit;

function isHierarchyUnit(token: AlignmentTimelineToken): token is AlignmentHierarchyUnit {
  return 'refinedStartSample' in token;
}

function tokenRefinedSpan(token: AlignmentTimelineToken): [number, number] {
  return isHierarchyUnit(token)
    ? [token.refinedStartSample, token.refinedEndSample]
    : [token.startSample, token.endSample];
}

function tokenLabel(token: AlignmentTimelineToken, level: AlignmentTimelineLane['level']): string {
  if (level === 'phoneme') return token.phoneme || token.text || '·';
  if (level === 'mora' && isHierarchyUnit(token)) return token.kana || token.text || '·';
  return token.text || token.phoneme || '·';
}

function tokenAnchor(token: AlignmentTimelineToken, refined: boolean): number {
  if (!isHierarchyUnit(token)) return token.startSample;
  return refined ? token.refinedSample : token.alignedSample;
}

function candidateStem(candidate: CandidateEvent): StemKind {
  return { vocals: 'vocals', melody: 'other', drums: 'drums', mix: 'mix' }[candidate.lane] as StemKind;
}

export interface AlignmentTimelineLane {
  method: AlignmentMethodId;
  label: string;
  color: string;
  level: AlignmentLayer | 'raw';
  tokens: AlignmentTimelineToken[];
}

interface TimelineCanvasProps {
  track: TrackDetail;
  initialWaveform?: WaveformPeaks;
  currentSample: number;
  isPlaying: boolean;
  followPlayback: boolean;
  onSeek: (sample: number) => void;
  mode?: 'edit' | 'alignment';
  alignmentLanes?: AlignmentTimelineLane[];
  waveformSources?: StemKind[];
  selectedAlignmentId?: string | null;
  onAlignmentSelect?: (token: AlignmentTimelineToken, lane: AlignmentTimelineLane) => void;
}

const CANVAS_TOKENS = {
  background: '--canvas-bg',
  ruler: '--canvas-ruler',
  focus: '--canvas-focus',
  lane: '--canvas-lane',
  laneAlternate: '--canvas-lane-alternate',
  alignmentLane: '--canvas-alignment-lane',
  divider: '--canvas-divider',
  dividerStrong: '--canvas-divider-strong',
  label: '--canvas-label',
  labelStrong: '--canvas-label-strong',
  labelSurface: '--canvas-label-surface',
  rulerTick: '--canvas-ruler-tick',
  gridBar: '--canvas-grid-bar',
  gridBeat: '--canvas-grid-beat',
  gridSubdivision: '--canvas-grid-subdivision',
  waveformCenter: '--canvas-waveform-center',
  connector: '--canvas-connector',
  alignmentOutline: '--canvas-alignment-outline',
  alignmentText: '--canvas-alignment-text',
  selectionFill: '--canvas-selection-fill',
  selectionStroke: '--canvas-selection-stroke',
  playhead: '--canvas-playhead',
  minimapWaveform: '--canvas-minimap-waveform',
  minimapViewport: '--canvas-minimap-viewport',
  minimapViewportBorder: '--canvas-minimap-viewport-border',
} satisfies Record<string, CssVariableName>;

interface CanvasColors {
  palette: Record<keyof typeof CANVAS_TOKENS, string>;
  stems: Record<StemKind, string>;
  bands: Record<HitPoint['band'], string>;
  candidates: Record<CandidateEvent['status'], string>;
  manualMarker: string;
  resolve: (value: string) => string;
}

function resolveCanvasColors(): CanvasColors {
  // Style calculation is captured once when the canvas mounts; paint frames
  // reuse the resolved palette even while the playhead is moving.
  const resolve = createCssColorResolver();
  const resolveRecord = <Key extends string>(record: Record<Key, string>) => Object.fromEntries(
    Object.entries(record).map(([key, value]) => [key, resolve(value as string)]),
  ) as Record<Key, string>;

  return {
    palette: resolveRecord(Object.fromEntries(
      Object.entries(CANVAS_TOKENS).map(([key, token]) => [key, cssVar(token)]),
    ) as Record<keyof typeof CANVAS_TOKENS, string>),
    stems: resolveRecord(Object.fromEntries(
      STEM_ORDER.map((source) => [source, cssVar(STEM_COLOR_TOKENS[source])]),
    ) as Record<StemKind, string>),
    bands: resolveRecord(BAND_COLORS),
    candidates: resolveRecord(CANDIDATE_COLORS),
    manualMarker: resolve(cssVar('--marker-manual')),
    resolve,
  };
}

interface TimelineInterval<T> {
  item: T;
  start: number;
  end: number;
}

interface TimelineIntervalIndex<T> {
  entries: TimelineInterval<T>[];
  maxSpan: number;
}

function buildIntervalIndex<T>(items: readonly T[], spanOf: (item: T) => readonly [number, number]): TimelineIntervalIndex<T> {
  let maxSpan = 0;
  const entries = items.flatMap((item) => {
    const span = spanOf(item);
    const start = Math.min(span[0], span[1]);
    const end = Math.max(span[0], span[1]);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
    maxSpan = Math.max(maxSpan, end - start);
    return [{ item, start, end }];
  }).sort((left, right) => left.start - right.start);
  return { entries, maxSpan };
}

function lowerBoundByStart<T>(entries: readonly TimelineInterval<T>[], sample: number): number {
  let low = 0;
  let high = entries.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (entries[middle].start < sample) low = middle + 1;
    else high = middle;
  }
  return low;
}

function intervalsOverlapping<T>(index: TimelineIntervalIndex<T>, startSample: number, endSample: number): TimelineInterval<T>[] {
  const first = lowerBoundByStart(index.entries, startSample - index.maxSpan);
  const afterLast = lowerBoundByStart(index.entries, endSample + 1);
  return index.entries.slice(first, afterLast).filter((entry) => entry.end >= startSample);
}

interface DragState { id: string; originX: number; originSample: number }
interface SelectionState { originX: number; currentX: number; additive: boolean }

function lodForZoom(pixelsPerSecond: number): number | 'auto' {
  if (pixelsPerSecond >= 1000) return 0;
  if (pixelsPerSecond >= 360) return 1;
  if (pixelsPerSecond >= 130) return 2;
  return 'auto';
}

function chooseRulerStep(pixelsPerSecond: number): number {
  return [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60].find((seconds) => seconds * pixelsPerSecond >= 74) ?? 120;
}

function evidenceSummary(point: HitPoint): string {
  const evidence = Object.entries(point.stemEvidence ?? {})
    .filter((entry): entry is [StemKind, number] => STEM_ORDER.includes(entry[0] as StemKind) && typeof entry[1] === 'number')
    .sort((left, right) => right[1] - left[1])
    .map(([source, strength]) => `${STEM_LABELS[source]} ${Math.round(Math.max(0, Math.min(1, strength)) * 100)}%`);
  return evidence.length ? evidence.join(' · ') : '暂无逐分轨证据';
}

export function TimelineCanvas({
  track,
  initialWaveform,
  currentSample,
  isPlaying,
  followPlayback,
  onSeek,
  mode = 'edit',
  alignmentLanes = EMPTY_ALIGNMENT_LANES,
  waveformSources,
  selectedAlignmentId = null,
  onAlignmentSelect,
}: TimelineCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLCanvasElement>(null);
  const minimapRef = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const didInitialFit = useRef(false);
  const [viewportWidth, setViewportWidth] = useState(900);
  const [pixelsPerSecond, setPixelsPerSecond] = useState(80);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const [hovered, setHovered] = useState<{ point: HitPoint; x: number } | null>(null);
  const [hoveredAlignment, setHoveredAlignment] = useState<{
    token: AlignmentTimelineToken;
    lane: AlignmentTimelineLane;
    x: number;
    top: number;
  } | null>(null);
  const canvasColors = useMemo(resolveCanvasColors, []);
  const hitPoints = useEditorStore((state) => state.hitPoints);
  const selectedIds = useEditorStore((state) => state.selectedIds);
  const tempo = useEditorStore((state) => state.tempoMap[0]);
  const subdivision = useEditorStore((state) => state.subdivision);
  const snapEnabled = useEditorStore((state) => state.snapEnabled);
  const filters = useEditorStore((state) => state.filters);
  const requestedVisibleStems = useEditorStore((state) => state.visibleStems);
  const activeStem = useEditorStore((state) => state.activeStem);
  const setActiveStem = useEditorStore((state) => state.setActiveStem);
  const selectOnly = useEditorStore((state) => state.selectOnly);
  const toggleSelection = useEditorStore((state) => state.toggleSelection);
  const selectMany = useEditorStore((state) => state.selectMany);
  const addHitPoint = useEditorStore((state) => state.addHitPoint);
  const beginPreview = useEditorStore((state) => state.beginPreview);
  const moveHitPreview = useEditorStore((state) => state.moveHitPreview);
  const commitPreview = useEditorStore((state) => state.commitPreview);
  const durationSec = track.sampleCount / track.originalSampleRate;
  const minPixelsPerSecond = Math.max(8, viewportWidth / Math.max(0.01, durationSec));
  const totalWidth = Math.max(viewportWidth, durationSec * pixelsPerSecond);
  const visibleHits = useMemo(() => filterHitPoints(hitPoints, filters), [filters, hitPoints]);
  const visibleCandidates = useMemo(
    () => (track.candidateEvents ?? []).filter(
      (candidate) => filters.showCandidateEvents
        && (filters.candidateLane === 'all' || candidate.lane === filters.candidateLane),
    ),
    [filters.candidateLane, filters.showCandidateEvents, track.candidateEvents],
  );
  const visibleSources = useMemo(
    () => resolveVisibleStemKinds(track.stems, waveformSources ?? requestedVisibleStems),
    [requestedVisibleStems, track.stems, waveformSources],
  );
  const activeAlignmentLanes = useMemo(
    () => mode === 'alignment' ? alignmentLanes : EMPTY_ALIGNMENT_LANES,
    [alignmentLanes, mode],
  );
  const visibleHitIndex = useMemo(
    () => buildIntervalIndex(visibleHits, (point) => [point.sample, point.sample]),
    [visibleHits],
  );
  const visibleCandidateIndex = useMemo(
    () => buildIntervalIndex(visibleCandidates, (candidate) => [candidate.acousticSample, candidate.chartSample]),
    [visibleCandidates],
  );
  const alignmentIndexes = useMemo(
    () => activeAlignmentLanes.map((lane) => buildIntervalIndex(lane.tokens, tokenRefinedSpan)),
    [activeAlignmentLanes],
  );
  const alignmentLaneColors = useMemo(
    () => activeAlignmentLanes.map((lane) => canvasColors.resolve(lane.color)),
    [activeAlignmentLanes, canvasColors],
  );
  const baseTimelineHeight = mode === 'alignment' ? ALIGNMENT_WAVEFORM_HEIGHT : EDIT_TIMELINE_HEIGHT;
  const timelineHeight = baseTimelineHeight + activeAlignmentLanes.length * ALIGNMENT_LANE_HEIGHT;
  const layout = useMemo(
    () => buildStemLaneLayout(visibleSources, baseTimelineHeight, RULER_HEIGHT),
    [baseTimelineHeight, visibleSources],
  );
  const laneBySource = useMemo(
    () => new Map(layout.lanes.map((lane) => [lane.source, lane])),
    [layout],
  );
  const showGrid = mode === 'alignment' || filters.showGrid;
  const showWaveform = mode === 'alignment' || filters.showWaveform;
  const gridVisibility = useMemo(
    () => gridVisibilityForScale(pixelsPerSecond, tempo?.bpm ?? 0, subdivision),
    [pixelsPerSecond, subdivision, tempo?.bpm],
  );
  const lodLevel = lodForZoom(pixelsPerSecond);
  const waveformQueries = useQueries({
    queries: STEM_ORDER.map((source) => ({
      queryKey: ['waveform', track.id, source, lodLevel],
      queryFn: async () => {
        const peaks = await api.getWaveform(track.id, lodLevel, source);
        return { ...peaks, source: peaks.source ?? source };
      },
      enabled: source === 'mix' || visibleSources.includes(source),
      placeholderData: source === 'mix' && initialWaveform
        ? { ...initialWaveform, source: initialWaveform.source ?? 'mix' as const }
        : undefined,
      staleTime: Infinity,
      retry: 1,
    })),
  });
  const waveforms = useMemo(() => {
    const entries: Partial<Record<StemKind, WaveformPeaks>> = {};
    STEM_ORDER.forEach((source, index) => {
      const waveform = waveformQueries[index]?.data;
      if (waveform) entries[source] = waveform;
    });
    return entries;
  }, [waveformQueries]);
  const overviewWaveform = waveforms.mix ?? initialWaveform;
  const availableSources = useMemo(() => new Set(resolveVisibleStemKinds(track.stems, STEM_ORDER)), [track.stems]);

  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const update = () => setViewportWidth(Math.max(320, element.clientWidth));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!didInitialFit.current && viewportWidth > 320) {
      didInitialFit.current = true;
      setPixelsPerSecond(Math.max(8, viewportWidth / Math.max(0.01, durationSec)));
    }
  }, [durationSec, viewportWidth]);

  useEffect(() => {
    if (!isPlaying || !followPlayback || !scrollRef.current) return;
    const playheadX = sampleToSeconds(currentSample, track.originalSampleRate) * pixelsPerSecond;
    const currentScroll = scrollRef.current.scrollLeft;
    if (playheadX > currentScroll + viewportWidth * 0.78 || playheadX < currentScroll + viewportWidth * 0.08) {
      scrollRef.current.scrollLeft = Math.max(0, playheadX - viewportWidth * 0.22);
    }
  }, [currentSample, followPlayback, isPlaying, pixelsPerSecond, track.originalSampleRate, viewportWidth]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tempo) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(viewportWidth * dpr);
    canvas.height = Math.round(timelineHeight * dpr);
    canvas.style.width = `${viewportWidth}px`;
    canvas.style.height = `${timelineHeight}px`;
    const context = canvas.getContext('2d');
    if (!context) return;
    const {
      palette,
      stems: stemColors,
      bands: bandColors,
      candidates: candidateColors,
      manualMarker,
    } = canvasColors;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, viewportWidth, timelineHeight);
    context.fillStyle = palette.background;
    context.fillRect(0, 0, viewportWidth, timelineHeight);
    context.fillStyle = palette.ruler;
    context.fillRect(0, 0, viewportWidth, RULER_HEIGHT);
    context.fillStyle = palette.focus;
    context.fillRect(0, layout.focusTop, viewportWidth, layout.focusBottom - layout.focusTop);

    const startSec = scrollLeft / pixelsPerSecond;
    const endSec = (scrollLeft + viewportWidth) / pixelsPerSecond;
    const startSample = Math.max(0, Math.floor(startSec * track.originalSampleRate));
    const endSample = Math.min(track.sampleCount - 1, Math.ceil(endSec * track.originalSampleRate));
    const sampleToX = (sample: number) => timelineXForSample(sample, track.originalSampleRate, pixelsPerSecond, scrollLeft);

    for (const [index, lane] of layout.lanes.entries()) {
      context.fillStyle = index % 2 === 0 ? palette.lane : palette.laneAlternate;
      context.fillRect(0, lane.top, viewportWidth, lane.height);
      context.strokeStyle = palette.divider;
      context.lineWidth = 1;
      context.beginPath(); context.moveTo(0, Math.round(lane.bottom) + 0.5); context.lineTo(viewportWidth, Math.round(lane.bottom) + 0.5); context.stroke();
    }

    if (mode === 'edit') {
      const activeLane = laneBySource.get(activeStem);
      if (activeLane) {
        context.save();
        context.globalAlpha = 0.075;
        context.fillStyle = stemColors[activeStem];
        context.fillRect(0, activeLane.top, viewportWidth, activeLane.height);
        context.globalAlpha = 0.58;
        context.fillRect(0, activeLane.top, 2, activeLane.height);
        context.restore();
      }
    }

    for (const [index] of activeAlignmentLanes.entries()) {
      const top = baseTimelineHeight + index * ALIGNMENT_LANE_HEIGHT;
      context.fillStyle = index % 2 === 0 ? palette.alignmentLane : palette.laneAlternate;
      context.fillRect(0, top, viewportWidth, ALIGNMENT_LANE_HEIGHT);
      context.strokeStyle = palette.dividerStrong;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(0, Math.round(top) + 0.5);
      context.lineTo(viewportWidth, Math.round(top) + 0.5);
      context.stroke();
    }

    context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.textBaseline = 'middle';
    for (const segment of track.focusMap ?? []) {
      if (segment.endSample <= startSample || segment.startSample > endSample) continue;
      const x1 = sampleToX(Math.max(startSample, segment.startSample));
      const x2 = sampleToX(Math.min(endSample, segment.endSample));
      const width = Math.max(1, x2 - x1);
      const source = STEM_ORDER.includes(segment.focusSource) ? segment.focusSource : 'mix';
      const isActiveSource = mode === 'edit' && source === activeStem;
      context.save();
      context.globalAlpha = 0.2
        + Math.max(0, Math.min(1, segment.confidence)) * 0.18
        + (isActiveSource ? 0.1 : 0);
      context.fillStyle = stemColors[source];
      context.fillRect(x1, layout.focusTop + 2, width, layout.focusBottom - layout.focusTop - 4);
      if (isActiveSource) {
        context.globalAlpha = 0.6;
        context.fillRect(x1, layout.focusBottom - 2, width, 1);
      }
      const lane = laneBySource.get(source);
      if (lane) {
        context.globalAlpha = 0.07 + Math.max(0, Math.min(1, segment.confidence)) * 0.07;
        context.fillRect(x1, lane.top, width, lane.height);
      }
      context.restore();
      if (width >= 72) {
        context.fillStyle = palette.labelStrong;
        context.fillText(`${STEM_LABELS[source]} ${Math.round(segment.confidence * 100)}%`, x1 + 5, (layout.focusTop + layout.focusBottom) / 2);
      }
    }
    context.strokeStyle = palette.dividerStrong;
    context.beginPath(); context.moveTo(0, layout.focusBottom + 0.5); context.lineTo(viewportWidth, layout.focusBottom + 0.5); context.stroke();
    context.fillStyle = palette.label;
    context.font = '7px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillText((track.focusMap ?? []).length ? 'FOCUS' : 'FOCUS · 暂无分段', 6, (layout.focusTop + layout.focusBottom) / 2);

    const rulerStep = chooseRulerStep(pixelsPerSecond);
    const firstRulerIndex = Math.floor(startSec / rulerStep);
    const lastRulerIndex = Math.ceil(endSec / rulerStep);
    context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.textBaseline = 'top';
    for (let index = firstRulerIndex; index <= lastRulerIndex; index += 1) {
      const seconds = index * rulerStep;
      const x = seconds * pixelsPerSecond - scrollLeft;
      context.strokeStyle = palette.rulerTick;
      context.beginPath(); context.moveTo(x + 0.5, 23); context.lineTo(x + 0.5, RULER_HEIGHT); context.stroke();
      context.fillStyle = palette.label;
      context.fillText(formatTime(seconds), x + 4, 6);
    }

    if (showGrid) {
      const lines = calculateGridLines(startSample, endSample, track.originalSampleRate, tempo, subdivision);
      for (const line of lines) {
        if (line.kind === 'subdivision' && !gridVisibility.showSubdivisions) continue;
        if (line.kind === 'beat' && !gridVisibility.showBeats) continue;
        const x = sampleToX(line.exactSample);
        context.beginPath();
        context.moveTo(Math.round(x) + 0.5, layout.lanesTop);
        context.lineTo(Math.round(x) + 0.5, timelineHeight);
        context.strokeStyle = line.kind === 'bar' ? palette.gridBar : line.kind === 'beat' ? palette.gridBeat : palette.gridSubdivision;
        context.lineWidth = line.kind === 'bar' ? 1.5 : 1;
        context.stroke();
        if (line.kind === 'bar' && line.barNumber && line.barNumber > 0) {
          context.fillStyle = palette.label;
          context.fillText(String(line.barNumber), x + 5, layout.lanesTop + 5);
        }
      }
    }

    if (showWaveform) {
      for (const lane of layout.lanes) {
        const waveform = waveforms[lane.source];
        if (!waveform?.mins?.length || !waveform.maxs?.length) continue;
        const amplitude = lane.height * 0.36;
        const firstPeak = Math.max(0, Math.floor(startSample / waveform.windowSize));
        const lastPeak = Math.min(waveform.mins.length - 1, Math.ceil(endSample / waveform.windowSize));
        const peakPixelWidth = (waveform.windowSize / track.originalSampleRate) * pixelsPerSecond;
        const groupSize = Math.max(1, Math.floor(0.8 / Math.max(0.0001, peakPixelWidth)));
        context.save();
        context.globalAlpha = lane.source === 'mix' ? 0.4 : 0.48;
        context.strokeStyle = stemColors[lane.source];
        context.lineWidth = Math.max(1, peakPixelWidth * groupSize);
        context.beginPath();
        for (let index = firstPeak; index <= lastPeak; index += groupSize) {
          let min = 0;
          let max = 0;
          const groupEnd = Math.min(lastPeak, index + groupSize - 1);
          for (let peakIndex = index; peakIndex <= groupEnd; peakIndex += 1) {
            min = Math.min(min, waveform.mins[peakIndex] ?? 0);
            max = Math.max(max, waveform.maxs[peakIndex] ?? 0);
          }
          const x = sampleToX(index * waveform.windowSize);
          context.moveTo(x, lane.center - Math.min(1, max) * amplitude);
          context.lineTo(x, lane.center - Math.max(-1, min) * amplitude);
        }
        context.stroke();
        context.restore();
        context.strokeStyle = palette.waveformCenter;
        context.lineWidth = 1;
        context.beginPath(); context.moveTo(0, lane.center + 0.5); context.lineTo(viewportWidth, lane.center + 0.5); context.stroke();
      }
    }

    for (const lane of layout.lanes) {
      const isActive = mode === 'edit' && lane.source === activeStem;
      context.fillStyle = palette.labelSurface;
      context.fillRect(5, lane.top + 5, 82, 17);
      if (isActive) {
        context.save();
        context.globalAlpha = 0.16;
        context.fillStyle = stemColors[lane.source];
        context.fillRect(5, lane.top + 5, 82, 17);
        context.globalAlpha = 0.72;
        context.strokeStyle = stemColors[lane.source];
        context.lineWidth = 1;
        context.strokeRect(5.5, lane.top + 5.5, 81, 16);
        context.restore();
      }
      context.fillStyle = stemColors[lane.source];
      context.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.textBaseline = 'middle';
      context.fillText(`${isActive ? '● ' : ''}${lane.source.toUpperCase()} · ${STEM_LABELS[lane.source]}`, 10, lane.top + 13.5);
    }

    if (mode === 'edit' && filters.showCandidateEvents) {
      for (const { item: candidate } of intervalsOverlapping(visibleCandidateIndex, startSample, endSample)) {
        if (
          Math.max(candidate.acousticSample, candidate.chartSample) < startSample
          || Math.min(candidate.acousticSample, candidate.chartSample) > endSample
        ) continue;
        const requestedSource = candidateStem(candidate);
        const source = availableSources.has(requestedSource) ? requestedSource : 'mix';
        const lane = laneBySource.get(source);
        if (!lane) continue;
        const acousticX = sampleToX(candidate.acousticSample);
        const chartX = sampleToX(candidate.chartSample);
        const markerY = lane.center + Math.min(12, lane.height * 0.18);
        const color = candidateColors[candidate.status];
        context.save();
        context.globalAlpha = candidate.status === 'accepted' ? 0.9 : candidate.status === 'uncertain' ? 0.72 : 0.38;
        context.strokeStyle = color;
        context.fillStyle = color;
        context.lineWidth = 1;
        if (candidate.acousticSample !== candidate.chartSample) {
          context.setLineDash([2, 3]);
          context.beginPath();
          context.moveTo(acousticX, markerY);
          context.lineTo(chartX, markerY - 10);
          context.stroke();
          context.setLineDash([]);
        }
        // Acoustic position: compact dot.
        context.beginPath();
        context.arc(acousticX, markerY, 2.25, 0, Math.PI * 2);
        context.fill();
        // Chart position: fine stem with a small dot, avoiding oversized
        // gameplay-style triangles while keeping the two positions distinct.
        const chartDotY = markerY - 10;
        context.beginPath();
        context.moveTo(Math.round(chartX) + 0.5, markerY - 16);
        context.lineTo(Math.round(chartX) + 0.5, markerY - 5);
        context.stroke();
        context.beginPath();
        context.arc(chartX, chartDotY, 2, 0, Math.PI * 2);
        context.fill();
        context.restore();
      }
    }

    if (mode === 'edit' && filters.showHitPoints) {
      const selected = new Set(selectedIds);
      for (const { item: point } of intervalsOverlapping(visibleHitIndex, startSample, endSample)) {
        if (point.acousticSample < startSample || point.acousticSample > endSample) continue;
        const requestedSource = primaryStemOf(point);
        const source = availableSources.has(requestedSource) ? requestedSource : 'mix';
        const lane = laneBySource.get(source);
        if (!lane) continue;
        const x = sampleToX(point.acousticSample);
        const suggestedX = sampleToX(point.chartSample);
        const color = point.source === 'manual'
          ? manualMarker
          : bandColors[point.band];
        if (point.acousticSample !== point.chartSample) {
          const connectorY = lane.top + 18;
          context.save();
          context.globalAlpha = selected.has(point.id) ? 0.48 : 0.25;
          context.strokeStyle = palette.connector;
          context.fillStyle = palette.connector;
          context.lineWidth = 1;
          context.setLineDash([2, 3]);
          context.beginPath();
          context.moveTo(Math.round(suggestedX) + 0.5, lane.top + 15);
          context.lineTo(Math.round(suggestedX) + 0.5, lane.bottom - 5);
          context.moveTo(x, connectorY + 0.5);
          context.lineTo(suggestedX, connectorY + 0.5);
          context.stroke();
          context.setLineDash([]);
          context.beginPath();
          context.arc(suggestedX, connectorY, 2.25, 0, Math.PI * 2);
          context.fill();
          context.restore();
        }
        if (selected.has(point.id)) {
          context.globalAlpha = 0.4;
          context.strokeStyle = color;
          context.lineWidth = 1;
          context.beginPath(); context.moveTo(Math.round(x) + 0.5, layout.lanesTop); context.lineTo(Math.round(x) + 0.5, timelineHeight); context.stroke();
        }
        context.globalAlpha = selected.has(point.id) ? 1 : 0.34 + point.confidence * 0.62;
        context.strokeStyle = color;
        context.lineWidth = selected.has(point.id) ? 2 : 1;
        context.beginPath(); context.moveTo(Math.round(x) + 0.5, lane.top + 15); context.lineTo(Math.round(x) + 0.5, lane.bottom - 5); context.stroke();
        context.fillStyle = color;
        context.beginPath(); context.arc(x, lane.top + 15, selected.has(point.id) ? 2.75 : 2.25, 0, Math.PI * 2); context.fill();
        if (point.locked) { context.font = '9px sans-serif'; context.fillText('◆', x - 4, lane.bottom - 13); }
      }
      context.globalAlpha = 1;
    }

    for (const [index, alignmentLane] of activeAlignmentLanes.entries()) {
      const top = baseTimelineHeight + index * ALIGNMENT_LANE_HEIGHT;
      const center = top + ALIGNMENT_LANE_HEIGHT / 2;
      const laneColor = alignmentLaneColors[index];
      for (const { item: token } of intervalsOverlapping(alignmentIndexes[index], startSample, endSample)) {
        const [refinedStartSample, refinedEndSample] = tokenRefinedSpan(token);
        if (
          !Number.isFinite(refinedStartSample)
          || !Number.isFinite(refinedEndSample)
          || refinedEndSample <= refinedStartSample
          || refinedEndSample < startSample
          || refinedStartSample > endSample
        ) continue;
        const startX = sampleToX(refinedStartSample);
        const endX = sampleToX(refinedEndSample);
        const width = Math.max(2, endX - startX);
        const confidence = Math.max(0, Math.min(1, token.confidence));
        const isHovered = hoveredAlignment?.token.id === token.id
          && hoveredAlignment.lane.method === alignmentLane.method;
        const isSelected = selectedAlignmentId === token.id;
        const emphasized = isHovered || isSelected;
        context.save();
        context.globalAlpha = emphasized ? 1 : 0.28 + confidence * 0.5;
        context.fillStyle = laneColor;
        if (alignmentLane.level === 'mora') {
          context.globalAlpha = emphasized ? 0.18 : 0.08 + confidence * 0.08;
          context.fillRect(startX, top + 3, width, ALIGNMENT_LANE_HEIGHT - 6);
          context.globalAlpha = emphasized ? 1 : 0.52 + confidence * 0.4;
          const markerX = sampleToX(tokenAnchor(token, true));
          context.strokeStyle = laneColor;
          context.lineWidth = emphasized ? 2 : 1;
          context.beginPath();
          context.moveTo(Math.round(markerX) + 0.5, top + 5);
          context.lineTo(Math.round(markerX) + 0.5, top + ALIGNMENT_LANE_HEIGHT - 4);
          context.stroke();
        } else {
          context.fillRect(startX, center - 8, width, 16);
          context.strokeStyle = emphasized ? palette.alignmentOutline : laneColor;
          context.lineWidth = emphasized ? 1.5 : 1;
          context.strokeRect(startX + 0.5, center - 7.5, Math.max(1, width - 1), 15);
        }
        context.beginPath();
        context.arc(startX, center, emphasized ? 3.5 : 2.5, 0, Math.PI * 2);
        context.fill();
        if (isHierarchyUnit(token)) {
          const alignedStartX = sampleToX(token.alignedStartSample);
          const alignedEndX = sampleToX(token.alignedEndSample);
          context.globalAlpha = 0.78;
          context.strokeStyle = palette.alignmentOutline;
          context.lineWidth = 1;
          context.setLineDash([2, 2]);
          context.strokeRect(
            alignedStartX + 0.5,
            center - 10.5,
            Math.max(1, alignedEndX - alignedStartX - 1),
            20,
          );
          context.setLineDash([]);
        }
        if (width >= 12) {
          context.globalAlpha = 1;
          context.fillStyle = alignmentLane.level === 'mora' ? palette.labelStrong : palette.alignmentText;
          context.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
          context.textBaseline = 'middle';
          context.save();
          context.beginPath();
          context.rect(startX + 2, center - 8, Math.max(0, width - 4), 16);
          context.clip();
          context.fillText(tokenLabel(token, alignmentLane.level), startX + 4, center + 0.5);
          context.restore();
        }
        if (isHierarchyUnit(token)) {
          const alignedX = sampleToX(tokenAnchor(token, false));
          const refinedX = sampleToX(tokenAnchor(token, true));
          context.globalAlpha = 0.72 + confidence * 0.28;
          context.strokeStyle = palette.alignmentOutline;
          context.lineWidth = 1;
          context.beginPath();
          context.moveTo(alignedX, center + 11);
          context.lineTo(refinedX, center + 11);
          context.stroke();
          context.fillStyle = palette.background;
          context.beginPath();
          context.arc(alignedX, center + 11, 2.5, 0, Math.PI * 2);
          context.fill();
          context.strokeStyle = laneColor;
          context.stroke();
          context.fillStyle = laneColor;
          context.beginPath();
          context.arc(refinedX, center + 11, 3, 0, Math.PI * 2);
          context.fill();
        }
        context.restore();
      }
      context.fillStyle = palette.labelSurface;
      context.fillRect(5, top + 7, 112, 24);
      context.fillStyle = laneColor;
      context.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.textBaseline = 'middle';
      context.fillText(alignmentLane.label.toUpperCase(), 10, center);
    }

    if (mode === 'edit' && selection && Math.abs(selection.currentX - selection.originX) > 3) {
      const left = Math.min(selection.originX, selection.currentX);
      const width = Math.abs(selection.currentX - selection.originX);
      context.fillStyle = palette.selectionFill; context.fillRect(left, layout.lanesTop, width, baseTimelineHeight - layout.lanesTop);
      context.strokeStyle = palette.selectionStroke; context.lineWidth = 1; context.setLineDash([4, 3]); context.strokeRect(left + 0.5, layout.lanesTop + 0.5, width, baseTimelineHeight - layout.lanesTop - 1); context.setLineDash([]);
    }

  }, [activeAlignmentLanes, activeStem, alignmentIndexes, alignmentLaneColors, availableSources, baseTimelineHeight, canvasColors, filters.showCandidateEvents, filters.showHitPoints, gridVisibility, hoveredAlignment, laneBySource, layout, mode, pixelsPerSecond, scrollLeft, selectedAlignmentId, selectedIds, selection, showGrid, showWaveform, subdivision, tempo, timelineHeight, track.focusMap, track.originalSampleRate, track.sampleCount, viewportWidth, visibleCandidateIndex, visibleHitIndex, waveforms]);

  useEffect(() => {
    const canvas = playheadRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(viewportWidth * dpr);
    const height = Math.round(timelineHeight * dpr);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvas.style.width = `${viewportWidth}px`;
    canvas.style.height = `${timelineHeight}px`;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, viewportWidth, timelineHeight);
    const playheadX = timelineXForSample(
      currentSample,
      track.originalSampleRate,
      pixelsPerSecond,
      scrollLeft,
    );
    if (playheadX < -2 || playheadX > viewportWidth + 2) return;
    context.strokeStyle = canvasColors.palette.playhead;
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(playheadX, 0);
    context.lineTo(playheadX, timelineHeight);
    context.stroke();
    context.fillStyle = canvasColors.palette.playhead;
    context.beginPath();
    context.moveTo(playheadX - 5, 0);
    context.lineTo(playheadX + 5, 0);
    context.lineTo(playheadX, 7);
    context.closePath();
    context.fill();
  }, [canvasColors.palette.playhead, currentSample, pixelsPerSecond, scrollLeft, timelineHeight, track.originalSampleRate, viewportWidth]);

  useEffect(() => {
    const canvas = minimapRef.current;
    if (!canvas || !overviewWaveform) return;
    const width = viewportWidth;
    const height = 54;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr; canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    const context = canvas.getContext('2d');
    if (!context) return;
    const palette = canvasColors.palette;
    context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height); context.fillStyle = palette.background; context.fillRect(0, 0, width, height);
    context.strokeStyle = palette.minimapWaveform; context.lineWidth = 1; context.beginPath();
    for (let x = 0; x < width; x += 1) {
      const start = Math.floor((x / width) * overviewWaveform.mins.length);
      const end = Math.max(start + 1, Math.floor(((x + 1) / width) * overviewWaveform.mins.length));
      let min = 0, max = 0;
      for (let index = start; index < Math.min(end, overviewWaveform.mins.length); index += 1) { min = Math.min(min, overviewWaveform.mins[index] ?? 0); max = Math.max(max, overviewWaveform.maxs[index] ?? 0); }
      context.moveTo(x + 0.5, 27 - max * 21); context.lineTo(x + 0.5, 27 - min * 21);
    }
    context.stroke();
    const viewportStart = scrollLeft / totalWidth * width;
    const viewportSize = Math.min(width, viewportWidth / totalWidth * width);
    context.fillStyle = palette.minimapViewport; context.fillRect(viewportStart, 0, viewportSize, height);
    context.strokeStyle = palette.minimapViewportBorder; context.strokeRect(viewportStart + 0.5, 0.5, Math.max(1, viewportSize - 1), height - 1);
  }, [canvasColors.palette, overviewWaveform, scrollLeft, totalWidth, viewportWidth]);

  const sampleAtX = (x: number) => Math.max(0, Math.min(track.sampleCount - 1, Math.round(((scrollLeft + x) / pixelsPerSecond) * track.originalSampleRate)));
  const xForSample = (sample: number) => timelineXForSample(sample, track.originalSampleRate, pixelsPerSecond, scrollLeft);
  const laneSourceForPoint = (point: HitPoint): StemKind => {
    const source = primaryStemOf(point);
    return availableSources.has(source) ? source : 'mix';
  };
  const laneAtY = (y: number) => layout.lanes.find(
    (lane, index) => y >= lane.top && (y < lane.bottom || (index === layout.lanes.length - 1 && y <= lane.bottom)),
  );
  const focusSegmentAtPosition = (x: number, y: number) => (
    y >= layout.focusTop && y < layout.focusBottom
      ? focusSegmentAtSample(track.focusMap, sampleAtX(x))
      : undefined
  );
  const stemAtPosition = (x: number, y: number): StemKind | undefined => {
    const laneSource = laneAtY(y)?.source;
    if (laneSource) return laneSource;
    const focusSource = focusSegmentAtPosition(x, y)?.focusSource;
    return focusSource && laneBySource.has(focusSource) ? focusSource : undefined;
  };
  const hitNearPosition = (x: number, y?: number) => {
    const sample = sampleAtX(x);
    const tolerance = Math.ceil((7 / pixelsPerSecond) * track.originalSampleRate);
    const nearby = intervalsOverlapping(visibleHitIndex, sample - tolerance, sample + tolerance);
    const pointerLane = y === undefined ? undefined : laneAtY(y);
    let nearest: HitPoint | undefined;
    let nearestScore = Number.POSITIVE_INFINITY;
    for (const { item: point } of nearby) {
      const lane = laneBySource.get(laneSourceForPoint(point));
      const distance = Math.abs(xForSample(point.sample) - x);
      if (!lane || distance > 7) continue;
      // A marker only owns hit space inside its actual lane. Different stems may
      // legitimately carry manual points at the exact same sample.
      if (y !== undefined && pointerLane?.source !== lane.source) continue;
      const score = distance;
      if (score < nearestScore) {
        nearest = point;
        nearestScore = score;
      }
    }
    return nearest;
  };
  const alignmentTokenNearPosition = (x: number, y: number) => {
    const laneIndex = Math.floor((y - baseTimelineHeight) / ALIGNMENT_LANE_HEIGHT);
    const lane = activeAlignmentLanes[laneIndex];
    if (!lane || y < baseTimelineHeight) return undefined;
    const index = alignmentIndexes[laneIndex];
    const sample = sampleAtX(x);
    const tolerance = Math.ceil((4 / pixelsPerSecond) * track.originalSampleRate);
    let nearest: AlignmentTimelineToken | undefined;
    let nearestSpan = Number.POSITIVE_INFINITY;
    for (const { item: token } of intervalsOverlapping(index, sample - tolerance, sample + tolerance)) {
      const [tokenStart, tokenEnd] = tokenRefinedSpan(token);
      if (tokenEnd <= tokenStart) continue;
      const left = xForSample(tokenStart);
      const right = Math.max(left + 2, xForSample(tokenEnd));
      const span = tokenEnd - tokenStart;
      if (x >= left - 4 && x <= right + 4 && span < nearestSpan) {
        nearest = token;
        nearestSpan = span;
      }
    }
    if (!nearest) return undefined;
    return {
      token: nearest,
      lane,
      x: xForSample(tokenRefinedSpan(nearest)[0]),
      top: baseTimelineHeight + laneIndex * ALIGNMENT_LANE_HEIGHT,
    };
  };

  const pointerPosition = (event: React.PointerEvent<HTMLCanvasElement> | React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    const { x, y } = pointerPosition(event);
    if (mode === 'alignment') {
      const alignment = alignmentTokenNearPosition(x, y);
      event.currentTarget.setPointerCapture(event.pointerId);
      dragRef.current = null;
      setSelection(null);
      setHovered(null);
      setHoveredAlignment(alignment ?? null);
      if (alignment) onAlignmentSelect?.(alignment.token, alignment.lane);
      return;
    }
    const pointerStem = stemAtPosition(x, y);
    if (pointerStem) setActiveStem(pointerStem);
    const point = filters.showHitPoints ? hitNearPosition(x, y) : undefined;
    event.currentTarget.setPointerCapture(event.pointerId);
    if (point) {
      if (event.metaKey || event.ctrlKey) toggleSelection(point.id); else if (!selectedIds.includes(point.id)) selectOnly(point.id);
      if (!point.locked) { beginPreview(); dragRef.current = { id: point.id, originX: x, originSample: point.sample }; }
      setSelection(null);
    } else {
      dragRef.current = null;
      if (!event.metaKey && !event.ctrlKey) selectOnly(null);
      setSelection({ originX: x, currentX: x, additive: event.metaKey || event.ctrlKey });
    }
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = pointerPosition(event);
    if (mode === 'alignment') {
      const alignment = alignmentTokenNearPosition(x, y);
      setHoveredAlignment(alignment ?? null);
      setHovered(null);
      event.currentTarget.style.cursor = alignment ? 'pointer' : 'crosshair';
      return;
    }
    if (dragRef.current) {
      const drag = dragRef.current;
      const deltaSamples = Math.round(((x - drag.originX) / pixelsPerSecond) * track.originalSampleRate);
      let targetSample = drag.originSample + deltaSamples;
      if (snapEnabled && tempo) targetSample = nearestGridSample(targetSample, track.originalSampleRate, tempo, subdivision);
      moveHitPreview(drag.id, targetSample);
      setHovered(null);
    } else if (selection) {
      setSelection({ ...selection, currentX: x });
    } else {
      const point = filters.showHitPoints ? hitNearPosition(x, y) : undefined;
      setHovered(point ? { point, x: xForSample(point.sample) } : null);
      const focusSegment = focusSegmentAtPosition(x, y);
      event.currentTarget.style.cursor = point
        ? (point.locked ? 'not-allowed' : 'ew-resize')
        : focusSegment && laneBySource.has(focusSegment.focusSource) ? 'pointer' : 'crosshair';
    }
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const { x } = pointerPosition(event);
    if (mode === 'alignment') {
      onSeek(sampleAtX(x));
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      return;
    }
    if (dragRef.current) { commitPreview(); dragRef.current = null; }
    else if (selection) {
      const distance = Math.abs(selection.currentX - selection.originX);
      if (distance > 4) {
        const minSample = sampleAtX(Math.min(selection.originX, selection.currentX));
        const maxSample = sampleAtX(Math.max(selection.originX, selection.currentX));
        const rangeIds = visibleHits.filter((point) => point.sample >= minSample && point.sample <= maxSample).map((point) => point.id);
        selectMany(selection.additive ? [...selectedIds, ...rangeIds] : rangeIds);
      } else onSeek(sampleAtX(x));
      setSelection(null);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    if (!scrollRef.current) return;
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      const x = event.nativeEvent.offsetX;
      const anchorSeconds = (scrollRef.current.scrollLeft + x) / pixelsPerSecond;
      const factor = Math.exp(-event.deltaY * 0.003);
      const next = Math.max(minPixelsPerSecond, Math.min(2200, pixelsPerSecond * factor));
      setPixelsPerSecond(next);
      requestAnimationFrame(() => { if (scrollRef.current) scrollRef.current.scrollLeft = Math.max(0, anchorSeconds * next - x); });
    } else if (event.shiftKey || Math.abs(event.deltaX) > 0) {
      event.preventDefault();
      scrollRef.current.scrollLeft += event.deltaX || event.deltaY;
    }
  };

  const setZoom = (next: number) => {
    const centerSeconds = (scrollLeft + viewportWidth / 2) / pixelsPerSecond;
    setPixelsPerSecond(next);
    requestAnimationFrame(() => { if (scrollRef.current) scrollRef.current.scrollLeft = Math.max(0, centerSeconds * next - viewportWidth / 2); });
  };

  const fit = () => { setPixelsPerSecond(minPixelsPerSecond); if (scrollRef.current) scrollRef.current.scrollLeft = 0; };
  const focusAtHover = hovered ? focusSegmentAtSample(track.focusMap, hovered.point.sample) : undefined;
  const hoverSnapErrorMs = hovered
    ? ((hovered.point.acousticSample - hovered.point.chartSample) / track.originalSampleRate) * 1000
    : 0;
  const loadingSources = STEM_ORDER.filter((source, index) => (source === 'mix' || visibleSources.includes(source)) && waveformQueries[index]?.isFetching);

  return (
    <section
      className="timeline-panel"
      ref={wrapRef}
      data-testid="timeline-panel"
      data-timeline-mode={mode}
      data-alignment-methods={activeAlignmentLanes.map((lane) => lane.method).join(',')}
      data-alignment-layers={activeAlignmentLanes.map((lane) => lane.level).join(',')}
      data-active-stem={activeStem}
    >
      <div className="timeline-canvas-wrap" style={{ height: `${timelineHeight}px` }}>
        <canvas
          ref={canvasRef}
          data-testid="timeline-canvas"
          aria-label={mode === 'alignment'
            ? '人声波形、BPM 网格及 Character、Mora、Phoneme 层级时间轴；空心点为 CTC 对齐，实心点为声学精修'
            : `分轨波形、soft focus、候选事件声学与游戏位置、BPM 网格及击打点时间轴；当前标记轨道：${STEM_LABELS[activeStem]}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => { setHovered(null); setHoveredAlignment(null); }}
          onLostPointerCapture={() => { setHovered(null); setHoveredAlignment(null); }}
          onPointerCancel={() => {
            useEditorStore.getState().cancelPreview();
            dragRef.current = null;
            setSelection(null);
            setHovered(null);
            setHoveredAlignment(null);
          }}
          onDoubleClick={mode === 'edit' ? (event) => {
            const { x, y } = pointerPosition(event);
            if (hitNearPosition(x, y)) return;
            const source = stemAtPosition(x, y) ?? activeStem;
            setActiveStem(source);
            addHitPoint(sampleAtX(x), source);
          } : undefined}
          onWheel={onWheel}
        />
        <canvas
          ref={playheadRef}
          className="timeline-playhead-canvas"
          aria-hidden="true"
        />
        {hovered ? <div className="hit-tooltip" role="tooltip" style={{ left: Math.max(10, Math.min(viewportWidth - 250, hovered.x + 10)) }}><strong>{hovered.point.source === 'manual' ? '手动击打点' : hovered.point.band}</strong><span>声学定位 {hovered.point.acousticSample.toLocaleString()} sample · {STEM_LABELS[laneSourceForPoint(hovered.point)]}</span><span>谱面位置 {hovered.point.chartSample.toLocaleString()} sample · 误差 {hoverSnapErrorMs >= 0 ? '+' : ''}{hoverSnapErrorMs.toFixed(2)} ms</span><span>{formatTime(hovered.point.timeSec)} · 置信度 {Math.round(hovered.point.confidence * 100)}%</span><span>{evidenceSummary(hovered.point)}</span>{focusAtHover ? <span>当前主导：{STEM_LABELS[focusAtHover.focusSource]}</span> : null}</div> : null}
        {hoveredAlignment ? (
          <div
            className="hit-tooltip alignment-token-tooltip"
            role="tooltip"
            style={{
              left: Math.max(10, Math.min(viewportWidth - 250, hoveredAlignment.x + 10)),
              top: Math.max(8, Math.min(timelineHeight - 132, hoveredAlignment.top + 4)),
            }}
          >
            <strong>{tokenLabel(hoveredAlignment.token, hoveredAlignment.lane.level)}</strong>
            <span>Character · {hoveredAlignment.token.text || '—'}</span>
            <span>Mora · {isHierarchyUnit(hoveredAlignment.token) ? hoveredAlignment.token.mora || hoveredAlignment.token.kana || '—' : '—'}</span>
            <span>Phoneme · {hoveredAlignment.token.phoneme || (hoveredAlignment.lane.level === 'phoneme' ? hoveredAlignment.token.text : '—')}</span>
            {isHierarchyUnit(hoveredAlignment.token) ? (
              <>
                <span>对齐 {hoveredAlignment.token.alignedStartSample.toLocaleString()} → {hoveredAlignment.token.alignedEndSample.toLocaleString()} sample</span>
                <span>精修 {hoveredAlignment.token.refinedStartSample.toLocaleString()} → {hoveredAlignment.token.refinedEndSample.toLocaleString()} sample</span>
                <span>锚点 {hoveredAlignment.token.alignedSample.toLocaleString()} → {hoveredAlignment.token.refinedSample.toLocaleString()} sample</span>
                {hoveredAlignment.token.evidence ? (
                  <span>
                    声学 Energy {Math.round(hoveredAlignment.token.evidence.energy * 100)}% · {' '}
                    Spectral {Math.round(hoveredAlignment.token.evidence.spectralChange * 100)}% · {' '}
                    Pitch {Math.round(hoveredAlignment.token.evidence.pitchChange * 100)}%
                  </span>
                ) : null}
                {hoveredAlignment.token.matchOperation ? (
                  <span>DP {hoveredAlignment.token.matchOperation}</span>
                ) : null}
              </>
            ) : (
              <>
                <span>{hoveredAlignment.token.startSample.toLocaleString()} → {hoveredAlignment.token.endSample.toLocaleString()} sample</span>
                <span>{formatTime(sampleToSeconds(hoveredAlignment.token.startSample, track.originalSampleRate))} → {formatTime(sampleToSeconds(hoveredAlignment.token.endSample, track.originalSampleRate))}</span>
              </>
            )}
            <span>Confidence · {Math.round(Math.max(0, Math.min(1, hoveredAlignment.token.confidence)) * 100)}%</span>
          </div>
        ) : null}
        {loadingSources.length ? <span className="lod-indicator">载入 {loadingSources.map((source) => STEM_LABELS[source]).join(' / ')} LOD {String(lodLevel)}…</span> : null}
        {showGrid && gridSubdivisionCount(subdivision) > 1 && !gridVisibility.showSubdivisions ? (
          <span className="grid-density-indicator">
            {gridVisibility.showBeats ? '细分线过密已隐藏' : '网格过密，仅显示小节线'}；{mode === 'alignment' ? '网格仅供节奏参照' : `${subdivision} 吸附仍有效`} · 放大查看
          </span>
        ) : null}
      </div>
      <div className="timeline-scrollbar" ref={scrollRef} onScroll={(event) => setScrollLeft(event.currentTarget.scrollLeft)}><div style={{ width: `${totalWidth}px` }} /></div>
      <div className="minimap-wrap"><canvas ref={minimapRef} aria-label="全局混音波形概览" onClick={(event) => { if (!scrollRef.current) return; const rect = event.currentTarget.getBoundingClientRect(); const ratio = (event.clientX - rect.left) / rect.width; scrollRef.current.scrollLeft = Math.max(0, ratio * totalWidth - viewportWidth / 2); }} /></div>
      <div className="timeline-zoom-controls">
        <button aria-label="缩小时间轴" onClick={() => setZoom(Math.max(minPixelsPerSecond, pixelsPerSecond / 2))}>−</button>
        <input aria-label="时间轴缩放" type="range" min={minPixelsPerSecond} max="1200" step="1" value={Math.min(1200, pixelsPerSecond)} onChange={(event) => setZoom(Number(event.target.value))} />
        <button aria-label="放大时间轴" onClick={() => setZoom(Math.min(1200, pixelsPerSecond * 2))}>＋</button>
        <output>{pixelsPerSecond < 100 ? pixelsPerSecond.toFixed(0) : Math.round(pixelsPerSecond)} px/s</output>
        <button onClick={fit}>适应全曲</button>
        {mode === 'edit' ? (
          <label className="marker-target-control">
            <span className={`stem-dot stem-${activeStem}`} />
            标记轨道
            <select aria-label="当前标记轨道" value={activeStem} onChange={(event) => setActiveStem(event.target.value as StemKind)}>
              {visibleSources.map((source) => <option key={source} value={source}>{STEM_LABELS[source]}</option>)}
            </select>
          </label>
        ) : null}
        <small>{mode === 'alignment' ? 'Ctrl/⌘ + 滚轮缩放 · Shift + 滚轮横移 · 点击试听' : 'Ctrl/⌘ + 滚轮缩放 · Shift + 滚轮横移 · 双击添加'}</small>
      </div>
    </section>
  );
}
