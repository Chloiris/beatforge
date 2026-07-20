import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '../src/api/client';
import { VocalLyricsPanel } from '../src/components/VocalLyricsPanel';
import type {
  AlignmentHierarchyUnit,
  AlignmentLayer,
  AlignmentResult,
  VocalLyrics,
} from '../src/types';

function lyrics(overrides: Partial<VocalLyrics> = {}): VocalLyrics {
  return {
    trackId: 'track-lyrics',
    text: '',
    inputFormat: 'japanese',
    status: 'empty',
    stage: 'idle',
    progress: 0,
    anchors: [],
    error: null,
    updatedAt: null,
    ...overrides,
  };
}

function unit(
  level: AlignmentLayer,
  index: number,
  text: string,
  mora: string | null,
  phoneme: string | null,
  refinedSample = 10_000 + index * 1_000,
): AlignmentHierarchyUnit {
  return {
    id: `${level}-${index}`,
    index,
    level,
    text,
    kana: mora,
    mora,
    phoneme,
    kind: level === 'phoneme' ? 'phone' : null,
    characterIndices: [],
    moraIndices: [],
    phonemeIndices: [],
    alignedStartSample: refinedSample - 50,
    alignedEndSample: refinedSample + 450,
    refinedStartSample: refinedSample,
    refinedEndSample: refinedSample + 500,
    alignedSample: refinedSample - 50,
    refinedSample,
    confidence: 0.94,
    observedTokenIndex: index,
    matchOperation: 'match',
    evidence: null,
  };
}

function hubertResult(status: AlignmentResult['status'] = 'completed'): AlignmentResult {
  const characters = [
    { ...unit('character', 0, '星', 'ホシ', 'h o sh i'), moraIndices: [0, 1], phonemeIndices: [0, 1] },
    { ...unit('character', 1, '火', 'ヒ', 'h i'), moraIndices: [2], phonemeIndices: [2, 3] },
  ];
  const moras = [
    { ...unit('mora', 0, '星', 'ホ', 'h o', 11_000), characterIndices: [0], phonemeIndices: [0, 1] },
    { ...unit('mora', 1, '星', 'シ', 'sh i', 12_000), characterIndices: [0], phonemeIndices: [1] },
    { ...unit('mora', 2, '火', 'ヒ', 'h i', 13_000), characterIndices: [1], phonemeIndices: [2, 3] },
  ];
  const phonemes = [
    { ...unit('phoneme', 0, '星', 'ホ', 'h', 11_100), characterIndices: [0], moraIndices: [0], phonemeIndices: [0] },
    { ...unit('phoneme', 1, '星', 'シ', 'sh', 11_300), characterIndices: [0], moraIndices: [0, 1], phonemeIndices: [1] },
    { ...unit('phoneme', 2, '火', 'ヒ', 'h', 13_100), characterIndices: [1], moraIndices: [2], phonemeIndices: [2] },
    { ...unit('phoneme', 3, '火', 'ヒ', 'i', 13_300), characterIndices: [1], moraIndices: [2], phonemeIndices: [3] },
  ];
  return {
    runId: 'ctc-run-2',
    trackId: 'track-lyrics',
    method: 'ctc',
    status,
    sampleRate: 44_100,
    sampleCount: 441_000,
    tokens: [],
    hierarchy: status === 'completed' ? { characters, moras, phonemes } : null,
    warnings: [],
    error: null,
    metadata: {},
    createdAt: '2026-07-19T00:00:00Z',
    updatedAt: '2026-07-19T00:00:01Z',
  };
}

function mockMissingHubertResult() {
  return vi.spyOn(api, 'getAlignmentResult').mockRejectedValue(
    new ApiError('missing', 'ALIGNMENT_RESULT_NOT_FOUND', 404),
  );
}

function renderPanel(node: ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return { client, ...render(<QueryClientProvider client={client}>{node}</QueryClientProvider>) };
}

describe('VocalLyricsPanel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('accepts Japanese lyrics and saves the selected input format', async () => {
    const user = userEvent.setup();
    mockMissingHubertResult();
    vi.spyOn(api, 'getVocalLyrics').mockResolvedValue(lyrics());
    const save = vi.spyOn(api, 'saveVocalLyrics').mockImplementation(
      async (trackId, text, inputFormat) => lyrics({
        trackId,
        text,
        inputFormat,
        status: 'saved',
        updatedAt: '2026-07-18T12:00:00Z',
      }),
    );

    renderPanel(<VocalLyricsPanel trackId="track-lyrics" defaultExpanded />);
    await waitFor(() => expect(api.getVocalLyrics).toHaveBeenCalledWith('track-lyrics'));

    await user.selectOptions(screen.getByLabelText('歌词输入格式'), 'lrc');
    await user.type(screen.getByLabelText('歌词文本'), '光を追いかけて');
    await user.click(screen.getByRole('button', { name: '仅保存歌词' }));

    await waitFor(() => {
      expect(save).toHaveBeenCalledWith('track-lyrics', '光を追いかけて', 'lrc');
    });
    expect(screen.getByRole('button', { name: '歌词已保存' })).toBeDisabled();
  });

  it('expands the typed Character → Mora → Phoneme hierarchy in explicit index order', async () => {
    const user = userEvent.setup();
    const onSeekSample = vi.fn();
    vi.spyOn(api, 'getVocalLyrics').mockResolvedValue(lyrics({
      text: '未来',
      status: 'saved',
    }));
    vi.spyOn(api, 'getAlignmentResult').mockResolvedValue(hubertResult());

    renderPanel(
      <VocalLyricsPanel
        trackId="track-lyrics"
        defaultExpanded
        onSeekSample={onSeekSample}
      />,
    );

    expect(await screen.findByText('2 Character · 3 Mora · 4 Phoneme')).toBeInTheDocument();
    const characterButtons = screen.getAllByRole('button', { name: /展开 Character/ });
    expect(characterButtons.map((button) => button.getAttribute('aria-label'))).toEqual([
      '展开 Character 1 星',
      '展开 Character 2 火',
    ]);
    expect(screen.queryByRole('button', { name: /展开 Mora/ })).not.toBeInTheDocument();

    await user.click(characterButtons[0]);
    const moraButtons = screen.getAllByRole('button', { name: /展开 Mora/ });
    expect(moraButtons.map((button) => button.getAttribute('aria-label'))).toEqual([
      '展开 Mora 1 ホ',
      '展开 Mora 2 シ',
    ]);

    await user.click(moraButtons[0]);
    const phonemeButtons = screen.getAllByRole('button', { name: /试听 Phoneme/ });
    expect(phonemeButtons.map((button) => button.getAttribute('aria-label'))).toEqual([
      '试听 Phoneme 1 h',
      '试听 Phoneme 2 sh',
    ]);
    expect(screen.getByText('11,100–11,600')).toBeInTheDocument();
    await user.click(phonemeButtons[0]);
    expect(onSeekSample).toHaveBeenCalledWith(11_100);
  });

  it('saves edits, starts fixed CTC, and polls the CTC result instead of a vocal alignment job', async () => {
    const user = userEvent.setup();
    vi.spyOn(api, 'getVocalLyrics').mockResolvedValue(lyrics({
      text: '光',
      status: 'saved',
      updatedAt: '2026-07-18T12:00:00Z',
    }));
    const getResult = vi.spyOn(api, 'getAlignmentResult')
      .mockRejectedValueOnce(new ApiError('missing', 'ALIGNMENT_RESULT_NOT_FOUND', 404))
      .mockResolvedValue(hubertResult('queued'));
    const run = vi.spyOn(api, 'runAlignment').mockResolvedValue(hubertResult('queued'));
    const legacyAlign = vi.spyOn(api, 'alignVocalLyrics');
    const getLegacyJob = vi.spyOn(api, 'getVocalLyricsJob');

    renderPanel(<VocalLyricsPanel trackId="track-lyrics" defaultExpanded />);
    await screen.findByDisplayValue('光');
    await user.click(screen.getByRole('button', { name: '运行 Japanese HuBERT CTC' }));

    await waitFor(() => expect(run).toHaveBeenCalledWith('track-lyrics', 'ctc'));
    await waitFor(() => expect(getResult.mock.calls.length).toBeGreaterThanOrEqual(2));
    expect(legacyAlign).not.toHaveBeenCalled();
    expect(getLegacyJob).not.toHaveBeenCalled();
    expect(screen.getByText('HuBERT 发音对齐').closest('li')).toHaveClass('active');
  });

  it('refreshes the project after the fixed HuBERT run completes', async () => {
    vi.spyOn(api, 'getVocalLyrics').mockResolvedValue(lyrics({ text: '光', status: 'saved' }));
    vi.spyOn(api, 'getAlignmentResult').mockResolvedValue(hubertResult());

    const { client } = renderPanel(
      <VocalLyricsPanel trackId="track-lyrics" defaultExpanded />,
    );
    const invalidate = vi.spyOn(client, 'invalidateQueries');

    await waitFor(() => {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ['project'] });
    });
  });

  it('keeps romaji save-only and never starts HuBERT', async () => {
    mockMissingHubertResult();
    vi.spyOn(api, 'getVocalLyrics').mockResolvedValue(lyrics({
      text: 'mirai wo hiraku',
      inputFormat: 'romaji',
      status: 'saved',
    }));
    const run = vi.spyOn(api, 'runAlignment');

    renderPanel(<VocalLyricsPanel trackId="track-lyrics" defaultExpanded />);
    await screen.findByDisplayValue('mirai wo hiraku');

    expect(screen.getByText(/罗马音仅保存/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '运行 Japanese HuBERT CTC' })).toBeDisabled();
    expect(run).not.toHaveBeenCalled();
  });
});
