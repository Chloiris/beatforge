import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ChartTransport } from '../src/components/ChartTransport';

describe('ChartTransport', () => {
  it('exposes play, pause-sized skips, and sample-accurate slider seeking', () => {
    const onTogglePlay = vi.fn();
    const onSeek = vi.fn();
    const onSeekBy = vi.fn();
    const onScrollSpeedChange = vi.fn();
    render(
      <ChartTransport
        currentTimeSec={18.25}
        durationSec={124.5}
        playing={false}
        scrollSpeed={4}
        onTogglePlay={onTogglePlay}
        onSeek={onSeek}
        onSeekBy={onSeekBy}
        onScrollSpeedChange={onScrollSpeedChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: '播放谱面预览' }));
    fireEvent.click(screen.getByRole('button', { name: '快退 5 秒' }));
    fireEvent.click(screen.getByRole('button', { name: '快进 5 秒' }));
    fireEvent.change(screen.getByRole('slider', { name: '播放位置' }), { target: { value: '42.125' } });
    fireEvent.change(screen.getByRole('combobox', { name: '谱面流速' }), { target: { value: '6' } });

    expect(onTogglePlay).toHaveBeenCalledOnce();
    expect(onSeekBy).toHaveBeenNthCalledWith(1, -5);
    expect(onSeekBy).toHaveBeenNthCalledWith(2, 5);
    expect(onSeek).toHaveBeenCalledWith(42.125);
    expect(onScrollSpeedChange).toHaveBeenCalledWith(6);
    expect(screen.getByRole('combobox', { name: '谱面流速' })).toHaveValue('4');
    expect(screen.getByText('0:18.250')).toBeVisible();
    expect(screen.getByText('/ 2:04.500')).toBeVisible();
  });
});
