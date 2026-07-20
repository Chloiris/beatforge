import { describe, expect, it } from 'vitest';
import { calculateGridLines, gridVisibilityForScale, nearestGridSample } from '../src/utils/grid';
import { sampleRate, tempo } from './fixtures';

describe('BPM grid', () => {
  it('recalculates positions immediately after BPM changes', () => {
    const at120 = calculateGridLines(0, sampleRate * 3, sampleRate, tempo, '1/4');
    const at128 = calculateGridLines(0, sampleRate * 3, sampleRate, { ...tempo, bpm: 128 }, '1/4');
    expect(at120[1].sample).toBe(22_050);
    expect(at128[1].sample).toBe(20_672);
    expect(at128[1].sample).not.toBe(at120[1].sample);
  });

  it('recalculates every line after offset changes', () => {
    const original = calculateGridLines(0, sampleRate * 2, sampleRate, tempo, '1/16');
    const shifted = calculateGridLines(0, sampleRate * 2, sampleRate, { ...tempo, beatOffsetSample: 321 }, '1/16');
    expect(shifted[0].sample).toBe(321);
    expect(shifted[1].sample - original[1].sample).toBe(321);
  });

  it('derives distant lines directly from beat index with no accumulated error', () => {
    const beatIndex = 100_000;
    const exact = tempo.beatOffsetSample + beatIndex * sampleRate * 60 / tempo.bpm;
    const lines = calculateGridLines(exact - 2, exact + 2, sampleRate, tempo, '1/32');
    expect(lines.find((line) => line.beatIndex === beatIndex && line.subdivisionIndex === 0)?.sample).toBe(Math.round(exact));
  });

  it('snaps using the full hidden subdivision grid', () => {
    expect(nearestGridSample(5_500, sampleRate, tempo, '1/16')).toBe(5_513);
  });

  it('uses the real subdivision spacing when deciding grid visibility', () => {
    const normalEditingZoom = gridVisibilityForScale(70, 129, '1/16');
    expect(normalEditingZoom.beatSpacingPixels).toBeCloseTo(32.56, 1);
    expect(normalEditingZoom.subdivisionSpacingPixels).toBeCloseTo(8.14, 1);
    expect(normalEditingZoom.showBeats).toBe(true);
    expect(normalEditingZoom.showSubdivisions).toBe(true);

    const fittedSongZoom = gridVisibilityForScale(30, 129, '1/16');
    expect(fittedSongZoom.showBeats).toBe(true);
    expect(fittedSongZoom.showSubdivisions).toBe(false);
  });
});
