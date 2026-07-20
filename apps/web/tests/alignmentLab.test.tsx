import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '../src/api/client';
import { AlignmentLab } from '../src/components/AlignmentLab';
import { useEditorStore } from '../src/state/editorStore';
import type {
  AlignmentHierarchyUnit,
  AlignmentLayer,
  AlignmentMethod,
  AlignmentReport,
  AlignmentResult,
  AlignmentResultStatus,
  CandidateEvent,
  TrackDetail,
  WaveformPeaks,
} from '../src/types';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

const methods: AlignmentMethod[] = [
  { id: 'qwen', name: 'Qwen Baseline', available: true },
  { id: 'mfa', name: 'MFA Japanese', available: true },
  { id: 'ctc', name: 'CTC Phoneme', available: true },
  { id: 'singing', name: 'Singing Alignment', available: false },
  { id: 'hybrid', name: 'Hybrid Fusion', available: true },
];
const track: TrackDetail = {
  id: 'track-alignment',
  projectId: 'project-alignment',
  createdAt: '2026-07-19T00:00:00.000Z',
  updatedAt: '2026-07-19T00:00:00.000Z',
  originalFileName: 'synthetic-vocal-demo.mp3',
  audioUrl: '/api/tracks/track-alignment/audio',
  format: 'mp3',
  originalSampleRate: sampleRate,
  channels: 2,
  sampleCount,
  durationSec: sampleCount / sampleRate,
  leadingSilenceSamples: 0,
  analysis: null,
  tempoMap: [tempo],
  hitPoints: [hit()],
  candidateEvents: [],
  stems: [
    { source: 'mix', available: true, waveformUrl: '/mix-waveform' },
    { source: 'vocals', available: true, waveformUrl: '/vocals-waveform' },
  ],
  focusMap: [],
  waveformUrl: '/mix-waveform',
};
const waveform: WaveformPeaks = {
  trackId: track.id,
  source: 'mix',
  sampleRate,
  sampleCount,
  level: 0,
  windowSize: sampleRate,
  mins: [-0.5, -0.7],
  maxs: [0.5, 0.7],
};

function unit(
  level: AlignmentLayer,
  index: number,
  text: string,
  mora: string | null,
  phoneme: string | null,
): AlignmentHierarchyUnit {
  const start = sampleRate * (index + 1);
  return {
    id: `${level}-${index}`,
    index,
    level,
    text,
    kana: mora,
    mora,
    phoneme,
    kind: level === 'phoneme' ? 'phone' : null,
    characterIndices: level === 'character' ? [index] : [Math.min(index, 1)],
    moraIndices: level === 'mora' ? [index] : [Math.min(index, 2)],
    phonemeIndices: level === 'phoneme' ? [index] : [index],
    alignedStartSample: start,
    alignedEndSample: start + 1_000,
    refinedStartSample: start + 40,
    refinedEndSample: start + 1_040,
    alignedSample: start,
    refinedSample: start + 40,
    confidence: 0.9,
    observedTokenIndex: index,
    matchOperation: 'match',
    evidence: null,
  };
}

function result(
  status: AlignmentResultStatus,
  withHierarchy = false,
  runId = 'ctc-run-1',
): AlignmentResult {
  return {
    runId,
    trackId: track.id,
    method: 'ctc',
    status,
    sampleRate,
    sampleCount,
    tokens: [],
    hierarchy: withHierarchy ? {
      characters: [
        { ...unit('character', 0, '星', 'ホシ', 'h o sh i'), moraIndices: [0, 1], phonemeIndices: [0, 1] },
        { ...unit('character', 1, '火', 'ヒ', 'h i'), moraIndices: [2], phonemeIndices: [2, 3] },
      ],
      moras: [
        { ...unit('mora', 0, '星', 'ホ', 'h o'), characterIndices: [0], phonemeIndices: [0] },
        { ...unit('mora', 1, '星', 'シ', 'sh i'), characterIndices: [0], phonemeIndices: [1] },
        { ...unit('mora', 2, '火', 'ヒ', 'h i'), characterIndices: [1], phonemeIndices: [2, 3] },
      ],
      phonemes: [
        unit('phoneme', 0, '星', 'ホ', 'h'),
        unit('phoneme', 1, '星', 'シ', 'sh'),
        unit('phoneme', 2, '火', 'ヒ', 'h'),
        unit('phoneme', 3, '火', 'ヒ', 'i'),
      ],
    } : null,
    warnings: [],
    error: status === 'failed' ? { code: 'MODEL_FAILED', message: 'HuBERT failed' } : null,
    metadata: {},
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
  };
}

function report(runId = 'ctc-run-1', score = 0.92): AlignmentReport {
  return {
    runId,
    trackId: track.id,
    method: 'ctc',
    score,
    coverage: score,
    acoustic: 0.87,
    rhythm: 0.9,
    stability: 0.95,
    lyricTokenCount: 2,
    alignedTokenCount: 2,
    details: {},
    createdAt: '2026-07-19T00:00:00.000Z',
  };
}

function moraCandidate(): CandidateEvent {
  return {
    id: 'candidate-mora-0',
    sample: sampleRate + 40,
    timeSec: (sampleRate + 40) / sampleRate,
    acousticSample: sampleRate + 40,
    chartSample: sampleRate + 400,
    snapErrorMs: -360 / sampleRate * 1_000,
    lane: 'vocals',
    sourceEvidence: { vocals: 1 },
    semanticEvidence: {},
    confidence: 0.84,
    status: 'accepted',
    gridType: 'straight_1_16',
    gridConfidence: 0.64,
    source: 'vocals',
    generator: 'hubert_ctc',
    character: '星',
    mora: 'ホ',
    phoneme: 'h',
    eventLevel: 'mora',
    eventPolicy: 'mora',
    alignmentUnitId: 'mora-event:mora-0',
    alignmentUnitIndex: 0,
    alignmentRunId: 'ctc-run-1',
    characterIndices: [0],
    phonemes: ['h'],
    alignedSample: sampleRate,
    refinedSample: sampleRate + 40,
    evidence: {
      hubert: 0.81,
      energy: 0.82,
      pitch: 0.83,
      rhythm: 0.84,
    },
    hitPointId: null,
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
  };
}

function renderLab(
  inputTrack: TrackDetail = track,
  onSeek = vi.fn(),
  onClose = vi.fn(),
) {
  useEditorStore.getState().initialize({
    trackId: inputTrack.id,
    signature: 'alignment-lab-test',
    sampleRate,
    sampleCount,
    hitPoints: inputTrack.hitPoints,
    tempoMap: inputTrack.tempoMap,
    availableStems: ['mix', 'vocals'],
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AlignmentLab
        track={inputTrack}
        initialWaveform={waveform}
        currentSample={0}
        isPlaying={false}
        followPlayback={false}
        onSeek={onSeek}
        onClose={onClose}
      />
    </QueryClientProvider>,
  );
}

describe('AlignmentLab', () => {
  afterEach(() => vi.restoreAllMocks());

  it('is fixed to HuBERT, defaults to Mora, and exposes layers in Character/Mora/Phoneme order', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(api, 'getAlignmentMethods').mockResolvedValue(methods);
    vi.spyOn(api, 'getAlignmentResult').mockResolvedValue(result('completed', true));
    vi.spyOn(api, 'getAlignmentReport').mockResolvedValue(report());

    renderLab(track, vi.fn(), onClose);
    const lab = await screen.findByTestId('alignment-lab');
    expect(lab).toHaveAttribute('id', 'vocal-alignment-panel');
    expect(lab).toHaveAccessibleName('Vocal Alignment');
    expect(within(lab).getByText('Vocal Alignment')).toBeInTheDocument();
    expect(within(lab).queryByText('SINGING ALIGNMENT LAB')).not.toBeInTheDocument();
    expect(within(lab).queryByRole('button', { name: '返回编辑器' })).not.toBeInTheDocument();
    const closeButton = within(lab).getByRole('button', { name: '关闭 Vocal Alignment' });
    expect(closeButton).toBeInTheDocument();

    const timeline = await screen.findByTestId('timeline-panel');
    await waitFor(() => expect(timeline).toHaveAttribute('data-alignment-layers', 'mora'));
    expect(timeline).toHaveAttribute('data-alignment-methods', 'ctc');

    expect(screen.queryByLabelText('Alignment 方法')).not.toBeInTheDocument();
    expect(screen.queryByRole('switch', { name: /Compare/ })).not.toBeInTheDocument();
    expect(screen.queryByText('Qwen Baseline')).not.toBeInTheDocument();
    expect(screen.queryByText('MFA Japanese')).not.toBeInTheDocument();
    expect(screen.queryByText('Hybrid Fusion')).not.toBeInTheDocument();

    const layerGroup = screen.getByRole('group', { name: 'Vocal Alignment Layer' });
    const layerButtons = within(layerGroup).getAllByRole('button');
    expect(layerButtons.map((button) => button.textContent)).toEqual([
      'Character字符',
      'Moraモーラ',
      'Phoneme音素',
    ]);
    expect(layerButtons[1]).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('2 characters · 3 moras · 4 phonemes')).toBeInTheDocument();
    expect(screen.getByText('3 Mora')).toBeInTheDocument();

    await user.click(layerButtons[0]);
    expect(timeline).toHaveAttribute('data-alignment-layers', 'character');
    expect(screen.getByText('2 Character')).toBeInTheDocument();
    await user.click(layerButtons[2]);
    expect(timeline).toHaveAttribute('data-alignment-layers', 'phoneme');
    expect(screen.getByText('4 Phoneme')).toBeInTheDocument();
    await user.click(closeButton);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('selects the first real Mora and shows exact CandidateEvent evidence', async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(api, 'getAlignmentMethods').mockResolvedValue(methods);
    vi.spyOn(api, 'getAlignmentResult').mockResolvedValue(result('completed', true));
    vi.spyOn(api, 'getAlignmentReport').mockResolvedValue(report());

    renderLab({ ...track, candidateEvents: [moraCandidate()] }, onSeek);

    const inspector = await screen.findByRole('complementary', { name: 'Mora Inspector' });
    expect(await within(inspector).findByText('candidate-mora-0')).toBeInTheDocument();
    expect(within(inspector).getByText('星')).toBeInTheDocument();
    expect(within(inspector).getByText('ホ')).toBeInTheDocument();
    expect(within(inspector).getByText('h')).toBeInTheDocument();
    const evidenceDetails = within(inspector).getByText('Evidence').closest('details');
    expect(evidenceDetails).not.toHaveAttribute('open');
    await user.click(within(inspector).getByText('Evidence'));
    expect(evidenceDetails).toHaveAttribute('open');
    const evidence = within(inspector).getByLabelText('HuBERT Mora evidence');
    for (const value of ['81%', '82%', '83%', '84%']) {
      expect(within(evidence).getByText(value)).toBeInTheDocument();
    }

    await user.click(within(inspector).getByRole('button', { name: '对齐' }));
    expect(onSeek).toHaveBeenCalledWith(sampleRate + 40);
    expect(within(inspector).getByRole('button', { name: '删除' })).toBeDisabled();
    expect(within(inspector).getByRole('button', { name: '锁定' })).toBeDisabled();
  });

  it('runs only CTC and drops the previous report while the new run is queued', async () => {
    const user = userEvent.setup();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(api, 'getAlignmentMethods').mockResolvedValue(methods);
    vi.spyOn(api, 'getAlignmentResult')
      .mockResolvedValueOnce(result('completed', true))
      .mockResolvedValue(result('queued', false, 'ctc-run-2'));
    vi.spyOn(api, 'getAlignmentReport').mockResolvedValue(report());
    const run = vi.spyOn(api, 'runAlignment').mockResolvedValue(
      result('queued', false, 'ctc-run-2'),
    );

    renderLab();
    expect((await screen.findAllByText('92%')).length).toBeGreaterThan(0);
    await user.click(screen.getByRole('button', { name: '运行 Japanese HuBERT CTC' }));

    await waitFor(() => expect(run).toHaveBeenCalledWith(track.id, 'ctc'));
    expect(await screen.findByText('本地任务运行中')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('92%')).not.toBeInTheDocument());
  });

  it('hides a report from a different completed run', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(api, 'getAlignmentMethods').mockResolvedValue(methods);
    vi.spyOn(api, 'getAlignmentResult').mockResolvedValue(result('completed', true, 'ctc-run-2'));
    vi.spyOn(api, 'getAlignmentReport').mockResolvedValue(report('ctc-run-1', 0.99));

    renderLab();
    await waitFor(() => expect(api.getAlignmentReport).toHaveBeenCalledWith(track.id, 'ctc'));
    expect(screen.queryByText('99%')).not.toBeInTheDocument();
    expect(screen.getByText('完成 HuBERT 对齐后显示评分。')).toBeInTheDocument();
  });

  it('treats a missing persisted HuBERT result as empty', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getWaveform').mockResolvedValue(waveform);
    vi.spyOn(api, 'getAlignmentMethods').mockResolvedValue(methods);
    vi.spyOn(api, 'getAlignmentResult').mockRejectedValue(
      new ApiError('missing', 'ALIGNMENT_RESULT_NOT_FOUND', 404),
    );
    const getReport = vi.spyOn(api, 'getAlignmentReport');

    renderLab();
    expect(await screen.findByText(/不会生成占位 timestamp/)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(getReport).not.toHaveBeenCalled();
  });
});
