import type { GridSubdivision, TempoSegment } from '../types';

export interface GridLine {
  sample: number;
  exactSample: number;
  beatIndex: number;
  subdivisionIndex: number;
  kind: 'bar' | 'beat' | 'subdivision';
  barNumber?: number;
}

export interface GridVisibility {
  beatSpacingPixels: number;
  subdivisionSpacingPixels: number;
  showBeats: boolean;
  showSubdivisions: boolean;
}

const SUBDIVISIONS_PER_BEAT: Record<GridSubdivision, number> = {
  '1/4': 1,
  '1/8': 2,
  '1/12': 3,
  '1/16': 4,
  '1/24': 6,
  '1/32': 8,
};

export function gridSubdivisionCount(subdivision: GridSubdivision): number {
  return SUBDIVISIONS_PER_BEAT[subdivision];
}

export function samplesPerBeat(sampleRate: number, bpm: number): number {
  return (sampleRate * 60) / bpm;
}

/**
 * Decides grid LOD from the distance between adjacent lines, rather than from
 * the distance between beats. This keeps a 1/16 grid visible at normal editing
 * zoom while still avoiding an unreadable solid fill when fully zoomed out.
 */
export function gridVisibilityForScale(
  pixelsPerSecond: number,
  bpm: number,
  subdivision: GridSubdivision,
): GridVisibility {
  if (!Number.isFinite(pixelsPerSecond) || pixelsPerSecond <= 0 || !Number.isFinite(bpm) || bpm <= 0) {
    return {
      beatSpacingPixels: 0,
      subdivisionSpacingPixels: 0,
      showBeats: false,
      showSubdivisions: false,
    };
  }
  const beatSpacingPixels = (60 / bpm) * pixelsPerSecond;
  const subdivisionSpacingPixels = beatSpacingPixels / gridSubdivisionCount(subdivision);
  return {
    beatSpacingPixels,
    subdivisionSpacingPixels,
    showBeats: beatSpacingPixels >= 4,
    showSubdivisions: subdivisionSpacingPixels >= 5,
  };
}

export function calculateGridLines(
  startSample: number,
  endSample: number,
  sampleRate: number,
  tempo: TempoSegment,
  subdivision: GridSubdivision,
): GridLine[] {
  if (tempo.bpm <= 0 || sampleRate <= 0 || endSample < startSample) return [];
  const beatSamples = samplesPerBeat(sampleRate, tempo.bpm);
  const subdivisions = gridSubdivisionCount(subdivision);
  const firstBeatIndex = Math.floor((startSample - tempo.beatOffsetSample) / beatSamples) - 1;
  const lastBeatIndex = Math.ceil((endSample - tempo.beatOffsetSample) / beatSamples) + 1;
  const result: GridLine[] = [];

  for (let beatIndex = firstBeatIndex; beatIndex <= lastBeatIndex; beatIndex += 1) {
    for (let subdivisionIndex = 0; subdivisionIndex < subdivisions; subdivisionIndex += 1) {
      // Every position is derived from its integer beat index. No previous position is accumulated.
      const exactSample =
        tempo.beatOffsetSample +
        beatIndex * beatSamples +
        (subdivisionIndex * beatSamples) / subdivisions;
      if (exactSample < startSample || exactSample > endSample || exactSample < 0) continue;
      const normalizedBeat = ((beatIndex % tempo.timeSignatureNumerator) + tempo.timeSignatureNumerator) % tempo.timeSignatureNumerator;
      const isBeat = subdivisionIndex === 0;
      const isBar = isBeat && normalizedBeat === 0;
      result.push({
        sample: Math.round(exactSample),
        exactSample,
        beatIndex,
        subdivisionIndex,
        kind: isBar ? 'bar' : isBeat ? 'beat' : 'subdivision',
        barNumber: isBar ? Math.floor(beatIndex / tempo.timeSignatureNumerator) + 1 : undefined,
      });
    }
  }
  return result;
}

export function nearestGridSample(
  sample: number,
  sampleRate: number,
  tempo: TempoSegment,
  subdivision: GridSubdivision,
): number {
  if (tempo.bpm <= 0 || sampleRate <= 0) return Math.round(sample);
  const step = samplesPerBeat(sampleRate, tempo.bpm) / gridSubdivisionCount(subdivision);
  const gridIndex = Math.round((sample - tempo.beatOffsetSample) / step);
  return Math.round(tempo.beatOffsetSample + gridIndex * step);
}

export function snapErrorMs(sample: number, snappedSample: number, sampleRate: number): number {
  return ((sample - snappedSample) / sampleRate) * 1000;
}
