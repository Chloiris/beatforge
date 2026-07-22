import { describe, expect, it } from 'vitest';
import {
  chartApproachSeconds,
  chartNoteY,
  clampPlaybackTime,
  flattenFivePanelEvents,
  visibleChartNotes,
} from '../src/utils/chartPreview';
import { referenceExcerptEvents } from './syntheticChartFixtures';

describe('five-lane chart preview math with synthetic events', () => {
  it('maps visual scroll speed to a shorter approach window without changing time', () => {
    expect(chartApproachSeconds(1)).toBe(2.6);
    expect(chartApproachSeconds(4)).toBe(0.65);
    expect(chartApproachSeconds(8)).toBe(0.325);
    expect(chartApproachSeconds(0)).toBe(2.6);
  });

  it('flattens taps, a hold, and a simultaneous jump without losing panel lanes', () => {
    const notes = flattenFivePanelEvents(referenceExcerptEvents);
    expect(notes).toHaveLength(4);
    expect(notes.map((note) => note.lane)).toEqual([0, 4, 0, 2]);
    expect(notes.find((note) => note.type === 'hold')).toMatchObject({
      startTimeSec: 2,
      endTimeSec: 2.5,
      lane: 4,
    });
  });

  it('places an event on the judgment line at its absolute parser time and future notes below it', () => {
    expect(chartNoteY(2, 2, 112, 444, 2.6)).toBe(112);
    expect(chartNoteY(2, 1.5, 112, 444, 2.6)).toBeGreaterThan(112);
    expect(chartNoteY(2, 2.25, 112, 444, 2.6)).toBeLessThan(112);
  });

  it('keeps a hold visible after its head crosses the judgment line until its tail passes', () => {
    const notes = flattenFivePanelEvents(referenceExcerptEvents);
    expect(visibleChartNotes(notes, 2.25, 2.6).some((note) => note.type === 'hold')).toBe(true);
    expect(visibleChartNotes(notes, 2.9, 2.6).some((note) => note.type === 'hold')).toBe(false);
  });

  it('clamps rewind and fast-forward to the configured audio duration', () => {
    expect(clampPlaybackTime(-5, 20)).toBe(0);
    expect(clampPlaybackTime(30, 20)).toBe(20);
    expect(clampPlaybackTime(6, 20)).toBe(6);
  });
});
