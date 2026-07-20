import type { ComponentProps } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '../src/api/client';
import { TimelineCanvas } from '../src/components/TimelineCanvas';
import { resetEditorStore, useEditorStore } from '../src/state/editorStore';
import type { TrackDetail, WaveformPeaks } from '../src/types';
import { availableStemKinds } from '../src/utils/stems';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

const point = hit({ sample: 0 });
const waveform: WaveformPeaks = {
  trackId: 'track-timeline',
  source: 'mix',
  sampleRate,
  sampleCount,
  level: 0,
  windowSize: sampleRate,
  mins: [-0.5, -0.7],
  maxs: [0.5, 0.7],
};
const track: TrackDetail = {
  id: 'track-timeline',
  projectId: 'project-timeline',
  createdAt: '2026-07-18T00:00:00.000Z',
  updatedAt: '2026-07-18T00:00:00.000Z',
  originalFileName: 'timeline.wav',
  audioUrl: '/api/tracks/track-timeline/audio',
  format: 'wav',
  originalSampleRate: sampleRate,
  channels: 2,
  sampleCount,
  durationSec: sampleCount / sampleRate,
  leadingSilenceSamples: 0,
  analysis: null,
  tempoMap: [tempo],
  hitPoints: [point],
  candidateEvents: [],
  stems: [{ source: 'mix', available: true, waveformUrl: '/waveform' }],
  focusMap: [],
  waveformUrl: '/waveform',
};

function renderTimeline(
  inputTrack: TrackDetail = track,
  props: Partial<ComponentProps<typeof TimelineCanvas>> = {},
) {
  useEditorStore.getState().initialize({
    trackId: inputTrack.id,
    signature: 'timeline-tooltip-test',
    sampleRate,
    sampleCount,
    hitPoints: inputTrack.hitPoints,
    tempoMap: [tempo],
    availableStems: availableStemKinds(inputTrack.stems),
  });
  useEditorStore.getState().updateFilters({
    band: 'all',
    stem: 'all',
    minConfidence: 0,
    onlyUnedited: false,
    onlyLowConfidence: false,
    onlyOffGrid: false,
    showGrid: true,
    showHitPoints: true,
    showWaveform: true,
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <TimelineCanvas
        track={inputTrack}
        initialWaveform={waveform}
        currentSample={0}
        isPlaying={false}
        followPlayback={false}
        onSeek={vi.fn()}
        {...props}
      />
    </QueryClientProvider>,
  );
}

describe('TimelineCanvas tooltip', () => {
  beforeEach(() => resetEditorStore());
  afterEach(() => vi.restoreAllMocks());

  it('arms a clicked waveform lane and allows same-sample markers on different stems', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const stemTrack: TrackDetail = {
      ...track,
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'vocals'] });
    useEditorStore.getState().updateFilters({ stem: 'mix', onlyUnedited: true });

    const canvas = screen.getByTestId('timeline-canvas');
    Object.defineProperty(canvas, 'setPointerCapture', { configurable: true, value: vi.fn() });
    // Edit canvas lanes begin at y=56. With two lanes, y=300 is vocals.
    fireEvent(canvas, new MouseEvent('pointerdown', { bubbles: true, button: 0, clientX: 0, clientY: 300 }));
    expect(useEditorStore.getState().activeStem).toBe('vocals');
    // A mix point already exists at x=0; it must not block the vocals lane.
    fireEvent.doubleClick(canvas, { clientX: 0, clientY: 300 });

    const state = useEditorStore.getState();
    const added = state.hitPoints.at(-1)!;
    expect(state.activeStem).toBe('vocals');
    expect(screen.getByTestId('timeline-panel')).toHaveAttribute('data-active-stem', 'vocals');
    expect(screen.getByLabelText('当前标记轨道')).toHaveValue('vocals');
    expect(added).toMatchObject({
      acousticSample: 0,
      source: 'manual',
      primaryStem: 'vocals',
      stemEvidence: { vocals: 1 },
    });
    expect(state.filters).toMatchObject({ stem: 'vocals', onlyUnedited: false });
    expect(state.hitPoints.filter((candidate) => candidate.acousticSample === 0)).toHaveLength(2);
  });

  it('selects a stem from both its label edge and empty waveform background', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const stemTrack: TrackDetail = {
      ...track,
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
        { source: 'drums', available: true, waveformUrl: '/waveform-drums' },
        { source: 'bass', available: true, waveformUrl: '/waveform-bass' },
        { source: 'other', available: true, waveformUrl: '/waveform-other' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'vocals', 'drums', 'bass', 'other'] });

    const canvas = screen.getByTestId('timeline-canvas');
    Object.defineProperty(canvas, 'setPointerCapture', { configurable: true, value: vi.fn() });

    // Five-track edit lanes occupy y=56..394. This is the vocals label edge.
    fireEvent(canvas, new MouseEvent('pointerdown', {
      bubbles: true, button: 0, clientX: 14, clientY: 150,
    }));
    expect(useEditorStore.getState().activeStem).toBe('vocals');

    // The same interaction must work on otherwise empty waveform/background space.
    fireEvent(canvas, new MouseEvent('pointerdown', {
      bubbles: true, button: 0, clientX: 280, clientY: 225,
    }));
    expect(useEditorStore.getState().activeStem).toBe('drums');
    expect(screen.getByTestId('timeline-panel')).toHaveAttribute('data-active-stem', 'drums');
    expect(screen.getByLabelText('当前标记轨道')).toHaveValue('drums');
  });

  it('selects the owning stem when clicking a colored Focus segment', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const stemTrack: TrackDetail = {
      ...track,
      focusMap: [{
        id: 'focus-other',
        startSample: 0,
        endSample: sampleCount,
        focusSource: 'other',
        confidence: 0.88,
        reason: 'melodic_lead',
        manuallyEdited: false,
      }],
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
        { source: 'other', available: true, waveformUrl: '/waveform-other' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'vocals', 'other'] });

    const canvas = screen.getByTestId('timeline-canvas');
    Object.defineProperty(canvas, 'setPointerCapture', { configurable: true, value: vi.fn() });
    // The Focus strip is the colored source-detection bar at y=34..56.
    fireEvent(canvas, new MouseEvent('pointerdown', {
      bubbles: true, button: 0, clientX: 180, clientY: 44,
    }));

    expect(useEditorStore.getState().activeStem).toBe('other');
    expect(screen.getByTestId('timeline-panel')).toHaveAttribute('data-active-stem', 'other');
    expect(screen.getByLabelText('当前标记轨道')).toHaveValue('other');
  });

  it('selects a marker owning lane before selecting the marker itself', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const vocalPoint = hit({
      id: 'vocal-marker',
      sample: 0,
      primaryStem: 'vocals',
      stemEvidence: { vocals: 0.96 },
    });
    const stemTrack: TrackDetail = {
      ...track,
      hitPoints: [vocalPoint],
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
        { source: 'drums', available: true, waveformUrl: '/waveform-drums' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'vocals', 'drums'] });
    useEditorStore.getState().setActiveStem('drums');

    const canvas = screen.getByTestId('timeline-canvas');
    Object.defineProperty(canvas, 'setPointerCapture', { configurable: true, value: vi.fn() });
    // With three lanes, the vocals lane spans approximately y=169..281.
    fireEvent(canvas, new MouseEvent('pointerdown', {
      bubbles: true, button: 0, clientX: 0, clientY: 220,
    }));

    expect(useEditorStore.getState()).toMatchObject({
      activeStem: 'vocals',
      selectedIds: ['vocal-marker'],
    });
  });

  it('uses only a low-alpha lane tint and keeps the five-track layout unchanged', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    let alpha = 1;
    const alphaStack: number[] = [];
    const fills: Array<{ x: number; y: number; width: number; height: number; alpha: number }> = [];
    const context = new Proxy(
      {
        fillRect: vi.fn((x: number, y: number, width: number, height: number) => {
          fills.push({ x, y, width, height, alpha });
        }),
        save: vi.fn(() => alphaStack.push(alpha)),
        restore: vi.fn(() => { alpha = alphaStack.pop() ?? 1; }),
      } as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
        set(target, property, value) {
          if (property === 'globalAlpha' && typeof value === 'number') alpha = value;
          target[property] = value;
          return true;
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
    const stemTrack: TrackDetail = {
      ...track,
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
        { source: 'drums', available: true, waveformUrl: '/waveform-drums' },
        { source: 'bass', available: true, waveformUrl: '/waveform-bass' },
        { source: 'other', available: true, waveformUrl: '/waveform-other' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'vocals', 'drums', 'bass', 'other'] });

    const panel = screen.getByTestId('timeline-panel');
    const canvas = screen.getByTestId('timeline-canvas') as HTMLCanvasElement;
    const canvasWrap = panel.querySelector<HTMLElement>('.timeline-canvas-wrap')!;
    Object.defineProperty(canvas, 'setPointerCapture', { configurable: true, value: vi.fn() });
    const originalLayout = {
      wrapHeight: canvasWrap.style.height,
      canvasHeight: canvas.style.height,
      pixelHeight: canvas.height,
      visibleStems: useEditorStore.getState().visibleStems,
    };
    fills.length = 0;

    // Bass lane center in a five-track layout.
    fireEvent(canvas, new MouseEvent('pointerdown', {
      bubbles: true, button: 0, clientX: 240, clientY: 292,
    }));

    await waitFor(() => expect(panel).toHaveAttribute('data-active-stem', 'bass'));
    expect({
      wrapHeight: canvasWrap.style.height,
      canvasHeight: canvas.style.height,
      pixelHeight: canvas.height,
      visibleStems: useEditorStore.getState().visibleStems,
    }).toEqual(originalLayout);

    const fullLaneSelectionFills = fills.filter(
      (fill) => fill.x === 0 && fill.width >= 300 && fill.height > 50 && fill.alpha < 1,
    );
    expect(fullLaneSelectionFills).toEqual([
      expect.objectContaining({ alpha: 0.075 }),
    ]);
  });

  it('uses the armed stem when a double click is outside waveform lanes', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const stemTrack: TrackDetail = {
      ...track,
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'drums', available: true, waveformUrl: '/waveform-drums' },
      ],
    };
    renderTimeline(stemTrack, { waveformSources: ['mix', 'drums'] });
    fireEvent.change(screen.getByLabelText('当前标记轨道'), { target: { value: 'drums' } });

    fireEvent.doubleClick(screen.getByTestId('timeline-canvas'), { clientX: 160, clientY: 12 });

    expect(useEditorStore.getState().hitPoints.at(-1)).toMatchObject({
      source: 'manual',
      primaryStem: 'drums',
      stemEvidence: { drums: 1 },
    });
  });

  it('clears the hit tooltip when the pointer leaves or is cancelled', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const { container } = renderTimeline();
    const canvas = screen.getByTestId('timeline-canvas');

    fireEvent(canvas, new MouseEvent('pointermove', { bubbles: true, clientX: 0, clientY: 100 }));
    expect(container.querySelector('.hit-tooltip')).toHaveTextContent('声学定位 0 sample');

    fireEvent.pointerLeave(canvas);
    expect(container.querySelector('.hit-tooltip')).not.toBeInTheDocument();

    fireEvent(canvas, new MouseEvent('pointermove', { bubbles: true, clientX: 0, clientY: 100 }));
    expect(container.querySelector('.hit-tooltip')).toBeInTheDocument();

    fireEvent.pointerCancel(canvas);
    expect(container.querySelector('.hit-tooltip')).not.toBeInTheDocument();
  });

  it('keeps mix and stem waveforms at low contrast', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    const alphaValues: number[] = [];
    const context = new Proxy(
      {} as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
        set(target, property, value) {
          if (property === 'globalAlpha' && typeof value === 'number') alphaValues.push(value);
          target[property] = value;
          return true;
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);

    renderTimeline({
      ...track,
      stems: [
        { source: 'mix', available: true, waveformUrl: '/waveform' },
        { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
      ],
    }, { waveformSources: ['mix', 'vocals'] });

    await waitFor(() => {
      expect(alphaValues).toEqual(expect.arrayContaining([0.4, 0.48]));
    });
  });

  it('draws candidates and hits as fine stems with compact dots instead of triangles', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    const context = new Proxy(
      {
        arc: vi.fn(),
        closePath: vi.fn(),
        lineTo: vi.fn(),
        moveTo: vi.fn(),
      } as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    const playheadContext = new Proxy(
      {} as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function getContext(this: HTMLCanvasElement) {
      return this.classList.contains('timeline-playhead-canvas') ? playheadContext : context;
    });
    renderTimeline({
      ...track,
      candidateEvents: [
        {
          id: 'candidate-1',
          sample: 100,
          timeSec: 100 / sampleRate,
          acousticSample: 100,
          chartSample: 200,
          snapErrorMs: (100 - 200) * 1_000 / sampleRate,
          lane: 'vocals',
          sourceEvidence: { vocals: 0.9 },
          semanticEvidence: { beatConfidence: 0.8 },
          confidence: 0.8,
          status: 'accepted',
          gridType: 'straight_1_16',
          gridConfidence: 0.8,
          hitPointId: null,
          createdAt: '2026-07-18T00:00:00.000Z',
          updatedAt: '2026-07-18T00:00:00.000Z',
        },
      ],
    });

    await waitFor(() => {
      expect(context.arc).toHaveBeenCalledWith(
        expect.any(Number),
        expect.any(Number),
        2,
        0,
        Math.PI * 2,
      );
    });
    expect(context.arc).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      2.25,
      0,
      Math.PI * 2,
    );
    const moveCalls = vi.mocked(context.moveTo).mock.calls;
    const lineCalls = vi.mocked(context.lineTo).mock.calls;
    expect(moveCalls.some(([moveX, moveY]) => lineCalls.some(
      ([lineX, lineY]) => moveX === lineX && Number(lineY) - Number(moveY) === 11,
    ))).toBe(true);
    expect(context.closePath).not.toHaveBeenCalled();
  });

  it('keeps Mora markers compact and exposes the professional hover fields', () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    renderTimeline(track, {
      mode: 'alignment',
      waveformSources: ['mix'],
      alignmentLanes: [{
        method: 'ctc',
        label: 'Japanese HuBERT CTC · Mora',
        color: 'var(--alignment)',
        level: 'mora',
        tokens: [{
          id: 'mora-0',
          index: 0,
          level: 'mora',
          text: '星',
          kana: 'ホ',
          mora: 'ホ',
          phoneme: 'h o',
          kind: null,
          characterIndices: [0],
          moraIndices: [0],
          phonemeIndices: [0, 1],
          alignedStartSample: 0,
          alignedEndSample: sampleRate,
          refinedStartSample: 0,
          refinedEndSample: sampleRate,
          alignedSample: 0,
          refinedSample: 0,
          confidence: 0.94,
          observedTokenIndex: 0,
          matchOperation: 'match',
          evidence: null,
        }],
      }],
    });

    const panel = screen.getByTestId('timeline-panel');
    expect(panel).toHaveAttribute('data-alignment-layers', 'mora');
    fireEvent(
      screen.getByTestId('timeline-canvas'),
      new MouseEvent('pointermove', { bubbles: true, clientX: 1, clientY: 210 }),
    );
    const tooltip = screen.getByRole('tooltip');
    expect(tooltip).toHaveTextContent('Character · 星');
    expect(tooltip).toHaveTextContent('Mora · ホ');
    expect(tooltip).toHaveTextContent('Phoneme · h o');
    expect(tooltip).toHaveTextContent('Confidence · 94%');
  });

  it('draws real alignment token intervals without enabling hit-point edits', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    const context = new Proxy(
      { fillRect: vi.fn(), fillText: vi.fn(), arc: vi.fn() } as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
    const addHitPoint = vi.spyOn(useEditorStore.getState(), 'addHitPoint');

    renderTimeline(track, {
      mode: 'alignment',
      waveformSources: ['mix'],
      alignmentLanes: [{
        method: 'qwen',
        label: 'Qwen Baseline',
        color: '#43d8d2',
        level: 'raw',
        tokens: [{
          id: 'qwen-token-1',
          text: 'あ',
          phoneme: 'a',
          startSample: sampleRate,
          endSample: sampleRate * 2,
          confidence: 0.8,
          method: 'qwen',
        }],
      }],
    });

    const panel = screen.getByTestId('timeline-panel');
    expect(panel).toHaveAttribute('data-timeline-mode', 'alignment');
    expect(panel).toHaveAttribute('data-alignment-methods', 'qwen');
    await waitFor(() => expect(context.fillText).toHaveBeenCalledWith(
      'あ',
      expect.any(Number),
      expect.any(Number),
    ));
    const tokenInterval = vi.mocked(context.fillRect).mock.calls.find((call) => call[3] === 16);
    expect(tokenInterval).toBeDefined();
    // The token starts at one second and lasts one second, so x and width share
    // the same sample-derived scale. No grid or anchor position is involved.
    expect(tokenInterval![0]).toBeCloseTo(tokenInterval![2], 5);

    fireEvent.doubleClick(screen.getByTestId('timeline-canvas'), { clientX: 80, clientY: 120 });
    expect(addHitPoint).not.toHaveBeenCalled();
  });

  it('renders hierarchy labels at refined spans and preserves aligned/refined anchors', async () => {
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    const context = new Proxy(
      { fillRect: vi.fn(), fillText: vi.fn(), arc: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn() } as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);

    renderTimeline(track, {
      mode: 'alignment',
      waveformSources: ['mix'],
      alignmentLanes: [{
        method: 'ctc',
        label: 'Japanese HuBERT CTC · Character',
        color: '#9f8cef',
        level: 'character',
        tokens: [{
          id: 'character-0',
          index: 0,
          level: 'character',
          text: '星',
          kana: 'ほし',
          mora: 'ほ',
          phoneme: 'h o',
          kind: null,
          characterIndices: [0],
          moraIndices: [0],
          phonemeIndices: [0],
          alignedStartSample: sampleRate,
          alignedEndSample: sampleRate * 2,
          refinedStartSample: sampleRate + 100,
          refinedEndSample: sampleRate * 2 + 100,
          alignedSample: sampleRate,
          refinedSample: sampleRate + 100,
          confidence: 0.94,
          observedTokenIndex: 0,
          matchOperation: 'match',
          evidence: null,
        }],
      }],
    });

    const panel = screen.getByTestId('timeline-panel');
    expect(panel).toHaveAttribute('data-alignment-layers', 'character');
    await waitFor(() => expect(context.fillText).toHaveBeenCalledWith(
      '星',
      expect.any(Number),
      expect.any(Number),
    ));
    expect(context.arc).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      2.5,
      0,
      Math.PI * 2,
    );
    expect(context.arc).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      3,
      0,
      Math.PI * 2,
    );
  });
});
