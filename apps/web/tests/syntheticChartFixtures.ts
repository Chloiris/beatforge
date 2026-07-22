import type { ChartDocument, ChartEvent, ChartNote } from '../src/types';

const tap = (lane: number): ChartNote => ({
  lane,
  type: 'tap',
  endTimeSec: null,
  endBeat: null,
  source: 'synthetic-test',
  confidence: 1,
  foot: null,
});

// Generated fixture: 120 BPM, zero offset, and deliberately simple event times.
// It is not copied or measured from any user song or reference corpus.
export const referenceExcerptEvents: ChartEvent[] = [
  {
    timeSec: 1,
    beat: 2,
    measure: 0,
    subdivision: 4,
    rowIndex: 2,
    notes: [tap(0)],
    sourceEventId: null,
    pattern: null,
  },
  {
    timeSec: 2,
    beat: 4,
    measure: 1,
    subdivision: 4,
    rowIndex: 0,
    notes: [{
      lane: 4,
      type: 'hold',
      endTimeSec: 2.5,
      endBeat: 5,
      source: 'synthetic-test',
      confidence: 1,
      foot: null,
    }],
    sourceEventId: null,
    pattern: null,
  },
  {
    timeSec: 6,
    beat: 12,
    measure: 3,
    subdivision: 4,
    rowIndex: 0,
    notes: [tap(0), tap(2)],
    sourceEventId: null,
    pattern: null,
  },
];

export const referenceExcerptChart: ChartDocument = {
  id: 'synthetic-reference-chart',
  title: 'Synthetic reference chart',
  artist: 'BeatForge Test Lab',
  music: 'synthetic-reference.wav',
  sourceGroup: 'SPEED_CLUB',
  sourcePath: 'SPEED_CLUB/synthetic-reference/synthetic-reference_Lv5.sm',
  mode: 'pump-single',
  laneCount: 5,
  difficulty: 'Hard',
  meter: 5,
  bpm: 120,
  offsetSec: 0,
  durationSec: 20,
  measureCount: 10,
  tempoMap: [{ beat: 0, bpm: 120, timeSec: 0 }],
  events: referenceExcerptEvents,
  statistics: null,
  validation: null,
  optimization: null,
  modelProvenance: null,
  generator: 'synthetic_test_fixture',
  generatorVersion: '1.0',
  seed: null,
};
