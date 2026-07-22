import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../src/api/client';
import { ChartGenerationPage } from '../src/pages/ChartGenerationPage';
import type { ChartGenerationResponse, GenerateChartRequest, ProjectDetail } from '../src/types';
import { hit, sampleRate, tempo } from './fixtures';
import { referenceExcerptChart } from './syntheticChartFixtures';

const trackId = 'track-generation';

const project: ProjectDetail = {
  id: 'project-generation',
  title: 'Regeneration Test',
  artist: 'BeatForge',
  genre: 'SPEED',
  coverUrl: '',
  status: 'completed',
  createdAt: '2026-07-21T00:00:00.000Z',
  updatedAt: '2026-07-21T00:00:00.000Z',
  trackId,
  track: {
    id: trackId,
    projectId: 'project-generation',
    createdAt: '2026-07-21T00:00:00.000Z',
    updatedAt: '2026-07-21T00:00:00.000Z',
    originalFileName: 'regeneration.mp3',
    audioUrl: `/api/tracks/${trackId}/audio`,
    format: 'mp3',
    originalSampleRate: sampleRate,
    channels: 2,
    sampleCount: Math.round(referenceExcerptChart.durationSec * sampleRate),
    durationSec: referenceExcerptChart.durationSec,
    leadingSilenceSamples: 0,
    analysis: null,
    tempoMap: [tempo],
    hitPoints: [hit()],
    candidateEvents: [],
    stems: [{ source: 'mix', available: true, waveformUrl: '/waveform' }],
    focusMap: [],
    waveformUrl: '/waveform',
  },
};

function response(index: number, seed: number): ChartGenerationResponse {
  const id = `generated-chart-${index}`;
  return {
    generationId: id,
    chart: { ...referenceExcerptChart, id, title: project.title, seed },
    referenceCorpus: {
      source: 'local reference corpus',
      chartCount: 12,
      songCount: 9,
      difficultyRange: [1, 15],
      model: { requested: true, available: true, used: true },
    },
  };
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/projects/project-generation/chart']}>
        <Routes>
          <Route path="/projects/:projectId/chart" element={<ChartGenerationPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe('ChartGenerationPage regeneration', () => {
  it('uses a fresh explicit seed, displays the new generation, and resets playback each time', async () => {
    const user = userEvent.setup();
    const requests: GenerateChartRequest[] = [];
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(api, 'getProject').mockResolvedValue(project);
    vi.spyOn(api, 'getLatestChart').mockResolvedValue({
      ...referenceExcerptChart,
      id: 'latest-chart',
      title: project.title,
    });
    vi.spyOn(api, 'generateChart').mockImplementation(async (_requestedTrackId, input) => {
      requests.push(input);
      return response(requests.length, input.seed!);
    });

    renderPage();

    const regenerate = await screen.findByRole('button', { name: '重新生成谱面' });
    const difficultySelect = screen.getByRole('combobox', { name: '谱面难度' });
    expect(difficultySelect).toHaveValue(String(referenceExcerptChart.meter));
    expect(difficultySelect.querySelectorAll('option')).toHaveLength(15);
    await user.selectOptions(difficultySelect, '11');
    await user.click(regenerate);
    await waitFor(() => expect(screen.getByRole('link', { name: '导出 SM' }))
      .toHaveAttribute('href', expect.stringContaining('generationId=generated-chart-1')));

    const firstPosition = screen.getByRole('slider', { name: '播放位置' });
    fireEvent.change(firstPosition, { target: { value: '12' } });
    expect((firstPosition as HTMLInputElement).value).toBe('12');

    await user.click(screen.getByRole('button', { name: '重新生成谱面' }));
    await waitFor(() => expect(screen.getByRole('link', { name: '导出 SM' }))
      .toHaveAttribute('href', expect.stringContaining('generationId=generated-chart-2')));

    expect(requests).toHaveLength(2);
    expect(requests[0]).toMatchObject({ difficulty: 11, enableSpin: false, useLocalModel: true });
    expect(requests[0]?.seed).toEqual(expect.any(Number));
    expect(requests[1]?.seed).toEqual(expect.any(Number));
    expect(requests[1]?.seed).not.toBe(requests[0]?.seed);
    expect((screen.getByRole('slider', { name: '播放位置' }) as HTMLInputElement).value).toBe('0');
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-current-time', '0.000');
  });
});
