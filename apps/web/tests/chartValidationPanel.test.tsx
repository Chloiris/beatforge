import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ChartValidationPanel } from '../src/components/ChartValidationPanel';
import { referenceExcerptChart } from './syntheticChartFixtures';

describe('ChartValidationPanel full-step result', () => {
  it('shows the strict alternating no-spin guarantee returned by the validator', () => {
    render(
      <ChartValidationPanel
        chart={{
          ...referenceExcerptChart,
          validation: {
            valid: true,
            score: 100,
            issues: [],
            metrics: { fullStepReachable: true },
          },
        }}
      />,
    );

    expect(screen.getByText('FULL STEP · 全步伐')).toBeInTheDocument();
    expect(screen.getByText('严格左右交替 · 无转圈')).toBeInTheDocument();
    expect(screen.getByText('通过')).toBeInTheDocument();
  });
});
