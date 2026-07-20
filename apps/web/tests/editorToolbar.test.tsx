import { MemoryRouter } from 'react-router-dom';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EditorToolbar } from '../src/components/EditorToolbar';
import { resetEditorStore, useEditorStore } from '../src/state/editorStore';
import type { CandidateEvent, ProjectDetail, TrackDetail } from '../src/types';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

function candidate(overrides: Partial<CandidateEvent> = {}): CandidateEvent {
  return {
    id: overrides.id ?? 'candidate-mora-1',
    sample: sampleRate,
    timeSec: 1,
    acousticSample: sampleRate,
    chartSample: sampleRate,
    snapErrorMs: 0,
    lane: 'vocals',
    sourceEvidence: { vocals: 1 },
    semanticEvidence: { lyricAlignment: 0.9 },
    confidence: 0.9,
    status: 'accepted',
    gridType: 'straight_1_16',
    gridConfidence: 0.8,
    source: 'vocals',
    generator: 'hubert_ctc',
    character: '星',
    mora: 'ほ',
    phoneme: 'h o',
    eventLevel: 'mora',
    eventPolicy: 'mora',
    alignmentUnitId: 'mora-1',
    alignmentUnitIndex: 0,
    alignmentRunId: 'ctc-run-1',
    characterIndices: [0],
    phonemes: ['h', 'o'],
    alignedSample: sampleRate,
    refinedSample: sampleRate,
    evidence: { hubert: 0.9, energy: 0.8, pitch: 0.7, rhythm: 0.8 },
    hitPointId: null,
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
    ...overrides,
  };
}

function track(): TrackDetail {
  return {
    id: 'track-toolbar',
    projectId: 'project-toolbar',
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
    originalFileName: '合成ボーカル.wav',
    audioUrl: '/api/tracks/track-toolbar/audio',
    format: 'wav',
    originalSampleRate: sampleRate,
    channels: 2,
    sampleCount,
    durationSec: sampleCount / sampleRate,
    leadingSilenceSamples: 0,
    analysis: null,
    tempoMap: [tempo],
    hitPoints: [hit()],
    candidateEvents: [
      candidate({ id: 'mora-1', confidence: 0.9 }),
      candidate({ id: 'mora-2', mora: 'つ', confidence: 0.92 }),
      candidate({
        id: 'legacy-character',
        eventLevel: 'character',
        eventPolicy: 'character',
        confidence: 1,
      }),
      candidate({
        id: 'non-vocal-mora',
        lane: 'melody',
        confidence: 1,
      }),
      candidate({
        id: 'legacy-generator-mora',
        generator: 'qwen',
        confidence: 1,
      }),
    ],
    stems: [{ source: 'mix', available: true, waveformUrl: '/waveform' }],
    focusMap: [],
    waveformUrl: '/waveform',
  };
}

function project(inputTrack: TrackDetail): ProjectDetail {
  return {
    id: 'project-toolbar',
    title: 'Synthetic Vocal Demo',
    artist: 'Demo Artist',
    genre: 'Rock',
    coverUrl: '',
    status: 'completed',
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
    trackId: inputTrack.id,
    track: inputTrack,
  };
}

function renderToolbar() {
  const inputTrack = track();
  const onToggleAlignmentLab = vi.fn();
  useEditorStore.getState().initialize({
    trackId: inputTrack.id,
    signature: 'toolbar-test',
    sampleRate,
    sampleCount,
    hitPoints: inputTrack.hitPoints,
    tempoMap: inputTrack.tempoMap,
    availableStems: ['mix'],
  });
  render(
    <MemoryRouter>
      <EditorToolbar
        project={project(inputTrack)}
        track={inputTrack}
        mode="balanced"
        sensitivity={0.5}
        onModeChange={vi.fn()}
        onSensitivityChange={vi.fn()}
        onAnalyze={vi.fn()}
        onShowGuide={vi.fn()}
        alignmentLabOpen={false}
        onToggleAlignmentLab={onToggleAlignmentLab}
        retrySave={vi.fn()}
      />
    </MemoryRouter>,
  );
  return { inputTrack, onToggleAlignmentLab };
}

describe('EditorToolbar', () => {
  beforeEach(() => resetEditorStore());

  it('renders the professional three-part header and aggregates real HuBERT Mora events', async () => {
    const user = userEvent.setup();
    const { inputTrack, onToggleAlignmentLab } = renderToolbar();

    const header = screen.getByRole('banner', { name: 'BeatForge Studio 编辑器' });
    const identity = within(header).getByRole('group', { name: '项目与歌曲' });
    expect(within(identity).getByRole('link', { name: 'BeatForge Studio 首页' })).toBeVisible();
    expect(within(identity).getByText('Synthetic Vocal Demo')).toBeInTheDocument();
    expect(within(identity).getByText(/合成ボーカル\.wav/)).toBeInTheDocument();

    const workStatus = within(header).getByRole('status', { name: 'AI 工作状态' });
    expect(workStatus).toHaveTextContent('AI Analysis Complete');
    expect(workStatus).toHaveTextContent('2 vocal events generated');
    expect(workStatus).toHaveTextContent('Confidence 91%');

    const actions = within(header).getByRole('group', { name: '项目操作' });
    expect(within(actions).getByRole('status', { name: '保存状态' })).toHaveTextContent('已保存');
    expect(within(actions).getByRole('link', { name: /数据 \+ 参考音频/ })).toHaveAttribute(
      'href',
      `/api/tracks/${inputTrack.id}/export?format=package&audio=reference`,
    );
    expect(within(actions).getByRole('link', { name: /仅数据包/ })).toHaveAttribute(
      'href',
      `/api/tracks/${inputTrack.id}/export?format=package&audio=none`,
    );
    expect(within(actions).getByRole('link', { name: /完整分轨包/ })).toHaveAttribute(
      'href',
      `/api/tracks/${inputTrack.id}/export?format=package&audio=full`,
    );
    expect(within(actions).getByRole('link', { name: '导出 JSON' })).toHaveAttribute(
      'href',
      `/api/tracks/${inputTrack.id}/export?format=json`,
    );
    expect(within(actions).getByText('设置')).toBeInTheDocument();

    const vocalAlignment = within(actions).getByRole('button', { name: 'Vocal Alignment' });
    expect(vocalAlignment).toHaveAttribute('aria-controls', 'vocal-alignment-panel');
    expect(vocalAlignment).toHaveAttribute('aria-pressed', 'false');
    await user.click(vocalAlignment);
    expect(onToggleAlignmentLab).toHaveBeenCalledOnce();
    expect(screen.getByRole('spinbutton', { name: 'BPM' })).toHaveValue(tempo.bpm);
  });

  it('opens settings without replacing the existing analysis controls', async () => {
    const user = userEvent.setup();
    renderToolbar();

    const settingsSummary = screen.getByText('设置');
    const settings = settingsSummary.closest('details');
    expect(settings).not.toHaveAttribute('open');
    await user.click(settingsSummary);
    expect(settings).toHaveAttribute('open');
    expect(screen.getByLabelText('分析模式')).toHaveValue('balanced');
    expect(screen.getByRole('button', { name: /重新分析整首歌曲/ })).toBeInTheDocument();
  });
});
