import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChartPlayer } from '../src/components/ChartPlayer';
import { referenceExcerptChart } from './syntheticChartFixtures';

afterEach(() => vi.restoreAllMocks());

describe('ChartPlayer scroll speed', () => {
  it('defaults to 4x visual speed and never changes the audio playback rate', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const { container } = render(
      <ChartPlayer
        chart={referenceExcerptChart}
        audioUrl="/fixtures/synthetic-reference.wav"
        durationSec={referenceExcerptChart.durationSec}
      />,
    );

    const audio = container.querySelector('audio');
    expect(audio).not.toBeNull();
    expect(audio?.playbackRate).toBe(1);
    expect(screen.getByRole('combobox', { name: '谱面流速' })).toHaveValue('4');
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-scroll-speed', '4');

    fireEvent.change(screen.getByRole('combobox', { name: '谱面流速' }), {
      target: { value: '8' },
    });

    expect(audio?.playbackRate).toBe(1);
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-scroll-speed', '8');
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-approach-seconds', '0.325');
  });
});
