import { create } from 'zustand';
import type {
  GridSubdivision,
  HitBand,
  HitDisplayFilters,
  HitPoint,
  SaveStatus,
  StemKind,
  TempoSegment,
} from '../types';
import { nearestGridSample, snapErrorMs } from '../utils/grid';
import { isStemKind, primaryStemOf, STEM_ORDER } from '../utils/stems';
import { clampSample, sampleToSeconds } from '../utils/time';

interface EditorSnapshot {
  hitPoints: HitPoint[];
  tempoMap: TempoSegment[];
}

interface EditorState {
  trackId: string | null;
  loadedSignature: string;
  sampleRate: number;
  sampleCount: number;
  hitPoints: HitPoint[];
  tempoMap: TempoSegment[];
  selectedIds: string[];
  past: EditorSnapshot[];
  future: EditorSnapshot[];
  previewSnapshot: EditorSnapshot | null;
  revision: number;
  savedRevision: number;
  saveStatus: SaveStatus;
  saveError: string | null;
  subdivision: GridSubdivision;
  snapEnabled: boolean;
  availableStems: StemKind[];
  visibleStems: StemKind[];
  activeStem: StemKind;
  filters: HitDisplayFilters;
  initialize: (input: {
    trackId: string;
    signature: string;
    sampleRate: number;
    sampleCount: number;
    hitPoints: HitPoint[];
    tempoMap: TempoSegment[];
    availableStems?: StemKind[];
  }) => void;
  setSubdivision: (subdivision: GridSubdivision) => void;
  setSnapEnabled: (enabled: boolean) => void;
  setStemVisible: (source: StemKind, visible: boolean) => void;
  setActiveStem: (source: StemKind) => void;
  updateFilters: (changes: Partial<HitDisplayFilters>) => void;
  selectOnly: (id: string | null) => void;
  toggleSelection: (id: string) => void;
  selectMany: (ids: string[]) => void;
  addHitPoint: (sample: number, stem?: StemKind) => HitPoint;
  deleteSelected: () => void;
  updateHitPoint: (id: string, changes: Partial<HitPoint>) => void;
  updateSelectedBand: (band: HitBand) => void;
  setSelectedLocked: (locked: boolean) => void;
  nudgeSelected: (deltaSamples: number) => void;
  snapSelected: () => void;
  beginPreview: () => void;
  moveHitPreview: (id: string, sample: number) => void;
  commitPreview: () => void;
  cancelPreview: () => void;
  updateTempo: (changes: Partial<TempoSegment>) => void;
  undo: () => void;
  redo: () => void;
  setSaveStatus: (status: SaveStatus, error?: string | null) => void;
  markSaved: (revision: number) => void;
}

const defaultFilters: HitDisplayFilters = {
  band: 'all',
  stem: 'all',
  minConfidence: 0,
  onlyUnedited: false,
  onlyLowConfidence: false,
  onlyOffGrid: false,
  showGrid: true,
  showHitPoints: true,
  showWaveform: true,
  showCandidateEvents: true,
  candidateLane: 'all',
};

function cloneHitPoints(hitPoints: HitPoint[]): HitPoint[] {
  return hitPoints.map((point) => ({
    ...point,
    detectorVotes: [...point.detectorVotes],
    stemEvidence: { ...point.stemEvidence },
  }));
}

function cloneTempoMap(tempoMap: TempoSegment[]): TempoSegment[] {
  return tempoMap.map((segment) => ({ ...segment }));
}

function snapshot(state: Pick<EditorState, 'hitPoints' | 'tempoMap'>): EditorSnapshot {
  return { hitPoints: cloneHitPoints(state.hitPoints), tempoMap: cloneTempoMap(state.tempoMap) };
}

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `hit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function withDerivedSample(
  point: HitPoint,
  sample: number,
  sampleRate: number,
  sampleCount: number,
  tempo: TempoSegment,
  subdivision: GridSubdivision,
): HitPoint {
  const nextSample = clampSample(sample, sampleCount);
  const snappedSample = clampSample(nearestGridSample(nextSample, sampleRate, tempo, subdivision), sampleCount);
  return {
    ...point,
    sample: nextSample,
    acousticSample: nextSample,
    timeSec: sampleToSeconds(nextSample, sampleRate),
    refinedSample: nextSample,
    chartSample: snappedSample,
    snappedSample,
    snapErrorMs: snapErrorMs(nextSample, snappedSample, sampleRate),
    manuallyEdited: true,
    updatedAt: new Date().toISOString(),
  };
}

function mutateWithHistory(
  state: EditorState,
  mutation: (state: EditorState) => Partial<EditorState>,
): Partial<EditorState> {
  const before = snapshot(state);
  return {
    ...mutation(state),
    past: [...state.past.slice(-99), before],
    future: [],
    revision: state.revision + 1,
    saveStatus: 'idle',
    saveError: null,
  };
}

function fallbackActiveStem(preferred: StemKind, visibleStems: readonly StemKind[]): StemKind {
  if (visibleStems.includes(preferred)) return preferred;
  return visibleStems.includes('mix') ? 'mix' : visibleStems[0] ?? 'mix';
}

function filtersIncludingNewPoint(filters: HitDisplayFilters, point: HitPoint): HitDisplayFilters {
  const next = { ...filters };
  next.showHitPoints = true;
  if (next.band !== 'all' && next.band !== 'manual' && next.band !== point.band) next.band = 'manual';
  if (next.stem !== 'all' && next.stem !== point.primaryStem) next.stem = point.primaryStem;
  if (next.minConfidence > point.confidence) next.minConfidence = point.confidence;
  if (next.onlyUnedited && point.manuallyEdited) next.onlyUnedited = false;
  if (next.onlyLowConfidence && point.confidence >= 0.5) next.onlyLowConfidence = false;
  if (next.onlyOffGrid && Math.abs(point.snapErrorMs) <= 25) next.onlyOffGrid = false;
  return next;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  trackId: null,
  loadedSignature: '',
  sampleRate: 44_100,
  sampleCount: 0,
  hitPoints: [],
  tempoMap: [],
  selectedIds: [],
  past: [],
  future: [],
  previewSnapshot: null,
  revision: 0,
  savedRevision: 0,
  saveStatus: 'idle',
  saveError: null,
  subdivision: '1/16',
  snapEnabled: true,
  availableStems: ['mix'],
  visibleStems: ['mix'],
  activeStem: 'mix',
  filters: defaultFilters,

  initialize: ({ trackId, signature, sampleRate, sampleCount, hitPoints, tempoMap, availableStems }) =>
    set((state) => {
      if (state.trackId === trackId && state.loadedSignature === signature) return state;
      const safeTempo = tempoMap.length
        ? cloneTempoMap(tempoMap)
        : [{
            id: newId(),
            startSample: 0,
            bpm: 120,
            timeSignatureNumerator: 4,
            timeSignatureDenominator: 4,
            beatOffsetSample: 0,
            confidence: 0,
            manuallyEdited: false,
          }];
      const safeAvailableStems = STEM_ORDER.filter(
        (source) => source === 'mix' || (availableStems ?? []).some((candidate) => candidate === source),
      );
      const newlyAvailable = safeAvailableStems.filter((source) => !state.availableStems.includes(source));
      const previousVisible = state.trackId === trackId
        ? STEM_ORDER.filter((source) => safeAvailableStems.includes(source) && (state.visibleStems.includes(source) || newlyAvailable.includes(source)))
        : safeAvailableStems;
      const normalizedHits = hitPoints.map((point) => {
        const primaryStem = primaryStemOf(point);
        const acousticSample = clampSample(
          point.acousticSample ?? point.refinedSample ?? point.sample,
          sampleCount,
        );
        const chartSample = clampSample(
          point.chartSample ?? point.snappedSample ?? acousticSample,
          sampleCount,
        );
        return {
          ...point,
          sample: acousticSample,
          acousticSample,
          chartSample,
          timeSec: sampleToSeconds(acousticSample, sampleRate),
          refinedSample: acousticSample,
          snappedSample: chartSample,
          primaryStem,
          stemEvidence: { ...(point.stemEvidence ?? { [primaryStem]: point.confidence }) },
          detectorVotes: [...(point.detectorVotes ?? [])],
        };
      });
      const visibleStems: StemKind[] = previousVisible.length ? previousVisible : ['mix'];
      const activeStem = state.trackId === trackId
        ? fallbackActiveStem(state.activeStem, visibleStems)
        : fallbackActiveStem('mix', visibleStems);
      return {
        trackId,
        loadedSignature: signature,
        sampleRate,
        sampleCount,
        hitPoints: normalizedHits,
        tempoMap: safeTempo,
        availableStems: safeAvailableStems,
        visibleStems,
        activeStem,
        selectedIds: [],
        past: [],
        future: [],
        previewSnapshot: null,
        revision: 0,
        savedRevision: 0,
        saveStatus: 'saved',
        saveError: null,
      };
    }),

  setSubdivision: (subdivision) => set({ subdivision }),
  setSnapEnabled: (snapEnabled) => set({ snapEnabled }),
  setStemVisible: (source, visible) =>
    set((state) => {
      if (!isStemKind(source) || !state.availableStems.includes(source)) return state;
      const next = new Set(state.visibleStems);
      if (visible) next.add(source); else next.delete(source);
      const visibleStems = STEM_ORDER.filter((candidate) => next.has(candidate));
      const safeVisibleStems: StemKind[] = visibleStems.length ? visibleStems : ['mix'];
      return {
        visibleStems: safeVisibleStems,
        activeStem: fallbackActiveStem(state.activeStem, safeVisibleStems),
      };
    }),
  setActiveStem: (source) =>
    set((state) => (
      isStemKind(source)
      && state.availableStems.includes(source)
      && state.visibleStems.includes(source)
        ? { activeStem: source }
        : state
    )),
  updateFilters: (changes) => set((state) => ({ filters: { ...state.filters, ...changes } })),
  selectOnly: (id) => set({ selectedIds: id ? [id] : [] }),
  toggleSelection: (id) =>
    set((state) => ({
      selectedIds: state.selectedIds.includes(id)
        ? state.selectedIds.filter((selectedId) => selectedId !== id)
        : [...state.selectedIds, id],
    })),
  selectMany: (selectedIds) => set({ selectedIds: [...new Set(selectedIds)] }),

  addHitPoint: (sample, stem) => {
    const state = get();
    const tempo = state.tempoMap[0];
    const now = new Date().toISOString();
    const safeSample = clampSample(sample, state.sampleCount);
    const snappedSample = nearestGridSample(safeSample, state.sampleRate, tempo, state.subdivision);
    const requestedStem = isStemKind(stem) ? stem : state.activeStem;
    const primaryStem = state.availableStems.includes(requestedStem) && state.visibleStems.includes(requestedStem)
      ? requestedStem
      : fallbackActiveStem(state.activeStem, state.visibleStems);
    const point: HitPoint = {
      id: newId(),
      sample: safeSample,
      acousticSample: safeSample,
      chartSample: snappedSample,
      timeSec: sampleToSeconds(safeSample, state.sampleRate),
      detectedSample: safeSample,
      refinedSample: safeSample,
      snappedSample,
      snapErrorMs: snapErrorMs(safeSample, snappedSample, state.sampleRate),
      band: 'mid_hit',
      confidence: 1,
      salience: 1,
      source: 'manual',
      primaryStem,
      stemEvidence: { [primaryStem]: 1 },
      detectorVotes: ['manual'],
      manuallyEdited: true,
      locked: false,
      createdAt: now,
      updatedAt: now,
    };
    set((current) => mutateWithHistory(current, () => ({
      hitPoints: [...current.hitPoints, point],
      selectedIds: [point.id],
      activeStem: primaryStem,
      filters: filtersIncludingNewPoint(current.filters, point),
    })));
    return point;
  },

  deleteSelected: () =>
    set((state) => {
      if (!state.selectedIds.length) return state;
      const selected = new Set(state.selectedIds);
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.filter((point) => !selected.has(point.id) || point.locked),
        selectedIds: state.selectedIds.filter((id) => state.hitPoints.find((point) => point.id === id)?.locked),
      }));
    }),

  updateHitPoint: (id, changes) =>
    set((state) => {
      const existing = state.hitPoints.find((point) => point.id === id);
      if (!existing) return state;
      const tempo = state.tempoMap[0];
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.map((point) => {
          if (point.id !== id) return point;
          const requestedStem = changes.primaryStem;
          const primaryStem = isStemKind(requestedStem) && state.availableStems.includes(requestedStem)
            ? requestedStem
            : primaryStemOf(point);
          let stemEvidence = changes.stemEvidence
            ? { ...changes.stemEvidence }
            : { ...point.stemEvidence };
          if (requestedStem !== undefined && primaryStem !== primaryStemOf(point) && changes.stemEvidence === undefined) {
            if (point.source === 'manual') {
              stemEvidence = { [primaryStem]: 1 };
            } else if (typeof stemEvidence[primaryStem] !== 'number') {
              stemEvidence[primaryStem] = Math.max(
                point.confidence,
                stemEvidence[primaryStemOf(point)] ?? 0,
              );
            }
          }
          let updated = {
            ...point,
            ...changes,
            primaryStem,
            stemEvidence,
            manuallyEdited: true,
            updatedAt: new Date().toISOString(),
          };
          if (changes.sample !== undefined) {
            updated = withDerivedSample(updated, changes.sample, state.sampleRate, state.sampleCount, tempo, state.subdivision);
          }
          return updated;
        }),
      }));
    }),

  updateSelectedBand: (band) =>
    set((state) => {
      if (!state.selectedIds.length) return state;
      const selected = new Set(state.selectedIds);
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.map((point) =>
          selected.has(point.id) && !point.locked
            ? { ...point, band, manuallyEdited: true, updatedAt: new Date().toISOString() }
            : point,
        ),
      }));
    }),

  setSelectedLocked: (locked) =>
    set((state) => {
      if (!state.selectedIds.length) return state;
      const selected = new Set(state.selectedIds);
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.map((point) =>
          selected.has(point.id)
            ? { ...point, locked, manuallyEdited: true, updatedAt: new Date().toISOString() }
            : point,
        ),
      }));
    }),

  nudgeSelected: (deltaSamples) =>
    set((state) => {
      if (!state.selectedIds.length || deltaSamples === 0) return state;
      const selected = new Set(state.selectedIds);
      const tempo = state.tempoMap[0];
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.map((point) =>
          selected.has(point.id) && !point.locked
            ? withDerivedSample(point, point.sample + deltaSamples, state.sampleRate, state.sampleCount, tempo, state.subdivision)
            : point,
        ),
      }));
    }),

  snapSelected: () =>
    set((state) => {
      if (!state.selectedIds.length) return state;
      const selected = new Set(state.selectedIds);
      const tempo = state.tempoMap[0];
      return mutateWithHistory(state, () => ({
        hitPoints: state.hitPoints.map((point) =>
          selected.has(point.id) && !point.locked
            ? withDerivedSample(
                point,
                nearestGridSample(point.sample, state.sampleRate, tempo, state.subdivision),
                state.sampleRate,
                state.sampleCount,
                tempo,
                state.subdivision,
              )
            : point,
        ),
      }));
    }),

  beginPreview: () => set((state) => ({ previewSnapshot: snapshot(state) })),
  moveHitPreview: (id, sample) =>
    set((state) => {
      const tempo = state.tempoMap[0];
      return {
        hitPoints: state.hitPoints.map((point) =>
          point.id === id && !point.locked
            ? withDerivedSample(point, sample, state.sampleRate, state.sampleCount, tempo, state.subdivision)
            : point,
        ),
      };
    }),
  commitPreview: () =>
    set((state) => {
      if (!state.previewSnapshot) return state;
      const before = state.previewSnapshot;
      const changed = before.hitPoints.some((point, index) => point.sample !== state.hitPoints[index]?.sample);
      return changed
        ? {
            past: [...state.past.slice(-99), before],
            future: [],
            previewSnapshot: null,
            revision: state.revision + 1,
            saveStatus: 'idle',
            saveError: null,
          }
        : { previewSnapshot: null };
    }),
  cancelPreview: () =>
    set((state) =>
      state.previewSnapshot
        ? { hitPoints: cloneHitPoints(state.previewSnapshot.hitPoints), previewSnapshot: null }
        : state,
    ),

  updateTempo: (changes) =>
    set((state) => {
      if (!state.tempoMap.length) return state;
      const nextTempo = { ...state.tempoMap[0], ...changes, manuallyEdited: true };
      return mutateWithHistory(state, () => ({
        tempoMap: state.tempoMap.map((segment, index) =>
          index === 0 ? nextTempo : segment,
        ),
        // A tempo edit never moves sample, but its non-authoritative snap suggestion
        // and displayed error must immediately reflect the new grid.
        hitPoints: state.hitPoints.map((point) => {
          const snappedSample = clampSample(
            nearestGridSample(point.sample, state.sampleRate, nextTempo, state.subdivision),
            state.sampleCount,
          );
          return {
            ...point,
            chartSample: snappedSample,
            snappedSample,
            snapErrorMs: snapErrorMs(point.sample, snappedSample, state.sampleRate),
          };
        }),
      }));
    }),

  undo: () =>
    set((state) => {
      const previous = state.past.at(-1);
      if (!previous) return state;
      return {
        hitPoints: cloneHitPoints(previous.hitPoints),
        tempoMap: cloneTempoMap(previous.tempoMap),
        past: state.past.slice(0, -1),
        future: [snapshot(state), ...state.future.slice(0, 99)],
        selectedIds: state.selectedIds.filter((id) => previous.hitPoints.some((point) => point.id === id)),
        revision: state.revision + 1,
        saveStatus: 'idle',
        saveError: null,
      };
    }),
  redo: () =>
    set((state) => {
      const next = state.future[0];
      if (!next) return state;
      return {
        hitPoints: cloneHitPoints(next.hitPoints),
        tempoMap: cloneTempoMap(next.tempoMap),
        past: [...state.past.slice(-99), snapshot(state)],
        future: state.future.slice(1),
        selectedIds: state.selectedIds.filter((id) => next.hitPoints.some((point) => point.id === id)),
        revision: state.revision + 1,
        saveStatus: 'idle',
        saveError: null,
      };
    }),
  setSaveStatus: (saveStatus, saveError = null) => set({ saveStatus, saveError }),
  markSaved: (revision) =>
    set((state) =>
      state.revision === revision
        ? { savedRevision: revision, saveStatus: 'saved', saveError: null }
        : state,
    ),
}));

export function filterHitPoints(
  hitPoints: HitPoint[],
  filters: HitDisplayFilters,
): HitPoint[] {
  return hitPoints.filter((point) => {
    if (filters.band !== 'all') {
      if (filters.band === 'manual' ? point.source !== 'manual' : point.band !== filters.band) return false;
    }
    if (filters.stem && filters.stem !== 'all' && primaryStemOf(point) !== filters.stem) return false;
    if (point.confidence < filters.minConfidence) return false;
    if (filters.onlyUnedited && point.manuallyEdited) return false;
    if (filters.onlyLowConfidence && point.confidence >= 0.5) return false;
    if (filters.onlyOffGrid && Math.abs(point.snapErrorMs) <= 25) return false;
    return true;
  });
}

export function resetEditorStore(): void {
  useEditorStore.setState({
    trackId: null,
    loadedSignature: '',
    sampleRate: 44_100,
    sampleCount: 0,
    hitPoints: [],
    tempoMap: [],
    selectedIds: [],
    past: [],
    future: [],
    previewSnapshot: null,
    revision: 0,
    savedRevision: 0,
    saveStatus: 'idle',
    saveError: null,
    subdivision: '1/16',
    snapEnabled: true,
    availableStems: ['mix'],
    visibleStems: ['mix'],
    activeStem: 'mix',
    filters: defaultFilters,
  });
}
