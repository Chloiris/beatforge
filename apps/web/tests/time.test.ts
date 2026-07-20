import { describe, expect, it } from 'vitest';
import { formatTime, sampleToSeconds, secondsToSample } from '../src/utils/time';

describe('sample/time conversion', () => {
  it('round-trips integer samples without cumulative drift', () => {
    for (const rate of [22_050, 44_100, 48_000, 96_000]) {
      for (const sample of [0, 1, 12_345, rate * 60 * 17 + 29]) {
        expect(secondsToSample(sampleToSeconds(sample, rate), rate)).toBe(sample);
      }
    }
  });

  it('formats time to millisecond precision', () => {
    expect(formatTime(65.1234)).toBe('1:05.123');
    expect(formatTime(-2)).toBe('0:00.000');
  });
});
