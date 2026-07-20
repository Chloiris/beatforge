import type { HitPoint, TempoSegment } from '../src/types';

export const sampleRate = 44_100;
export const sampleCount = sampleRate * 30;

export const tempo: TempoSegment = {
  id: 'tempo-1', startSample: 0, bpm: 120, timeSignatureNumerator: 4,
  timeSignatureDenominator: 4, beatOffsetSample: 0, confidence: 0.9, manuallyEdited: false,
};

export function hit(overrides: Partial<HitPoint> = {}): HitPoint {
  const sample = overrides.sample ?? 44_100;
  const acousticSample = overrides.acousticSample ?? sample;
  const chartSample = overrides.chartSample ?? overrides.snappedSample ?? sample;
  return {
    id: overrides.id ?? 'hit-1', sample, timeSec: sample / sampleRate,
    detectedSample: sample, refinedSample: sample, snappedSample: sample, snapErrorMs: 0,
    band: 'low_hit', confidence: 0.9, salience: 0.8, source: 'fused',
    primaryStem: 'mix', stemEvidence: { mix: 0.9 },
    detectorVotes: ['mix_flux', 'low_band'], manuallyEdited: false, locked: false,
    createdAt: '2026-07-18T00:00:00.000Z', updatedAt: '2026-07-18T00:00:00.000Z',
    ...overrides,
    acousticSample,
    chartSample,
  };
}
