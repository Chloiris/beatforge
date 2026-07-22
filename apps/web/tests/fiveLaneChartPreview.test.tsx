import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FiveLaneChartPreview } from '../src/components/FiveLaneChartPreview';
import { referenceExcerptChart } from './syntheticChartFixtures';

afterEach(() => vi.restoreAllMocks());

describe('FiveLaneChartPreview', () => {
  it('renders five labeled gameplay lanes and the fixture hold on a Canvas', async () => {
    const context = new Proxy(
      {
        fillText: vi.fn(),
        fillRect: vi.fn(),
        lineTo: vi.fn(),
        moveTo: vi.fn(),
        arc: vi.fn(),
      } as Record<PropertyKey, unknown>,
      {
        get(target, property) {
          if (!(property in target)) target[property] = vi.fn();
          return target[property];
        },
      },
    ) as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);

    render(<FiveLaneChartPreview chart={referenceExcerptChart} currentTimeSec={2.5} scrollSpeed={4} />);

    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-current-time', '2.500');
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-scroll-speed', '4');
    expect(screen.getByTestId('five-lane-preview')).toHaveAttribute('data-approach-seconds', '0.650');
    expect(screen.getByLabelText(/五轨谱面预览：Synthetic reference chart/)).toBeVisible();
    await waitFor(() => {
      for (const label of ['左下', '左上', '中心', '右上', '右下']) {
        expect(context.fillText).toHaveBeenCalledWith(label, expect.any(Number), 28);
      }
    });
    expect(context.fillRect).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
    );
  });

  it('matches a layout-constrained preview height instead of forcing the legacy fixed height', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function getWidth(
      this: HTMLElement,
    ) {
      return this instanceof HTMLCanvasElement ? 300 : 720;
    });
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function getHeight(
      this: HTMLElement,
    ) {
      return this instanceof HTMLCanvasElement ? 150 : 420;
    });

    render(<FiveLaneChartPreview chart={referenceExcerptChart} currentTimeSec={2.5} />);

    await waitFor(() => {
      expect(screen.getByTestId('five-lane-chart-canvas')).toHaveStyle({
        width: '720px',
        height: '420px',
      });
    });
  });
});
