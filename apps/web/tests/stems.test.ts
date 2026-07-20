import { beforeEach, describe, expect, it } from 'vitest';
import { filterHitPoints, resetEditorStore, useEditorStore } from '../src/state/editorStore';
import type { FocusSegment, StemDescriptor } from '../src/types';
import {
  buildStemLaneLayout,
  focusSegmentAtSample,
  resolveVisibleStemKinds,
  timelineXForSample,
} from '../src/utils/stems';
import { hit, sampleRate } from './fixtures';

describe('stem timeline geometry', () => {
  it('gives every lane a non-overlapping range while sharing one sample-to-x transform', () => {
    const layout = buildStemLaneLayout(['mix', 'vocals', 'drums', 'bass', 'other']);
    expect(layout.lanes).toHaveLength(5);
    for (let index = 1; index < layout.lanes.length; index += 1) {
      expect(layout.lanes[index].top).toBe(layout.lanes[index - 1].bottom);
      expect(layout.lanes[index].top).toBeGreaterThanOrEqual(layout.lanesTop);
    }
    expect(layout.lanes.at(-1)?.bottom).toBe(layout.height);

    const sample = 123_456;
    const expectedX = timelineXForSample(sample, sampleRate, 240, 70);
    for (const lane of layout.lanes) {
      expect(timelineXForSample(sample, sampleRate, 240, 70)).toBe(expectedX);
      expect(lane.center).toBeGreaterThan(lane.top);
      expect(lane.center).toBeLessThan(lane.bottom);
    }
  });

  it('uses start-inclusive and end-exclusive focus segment boundaries', () => {
    const segments: FocusSegment[] = [{
      id: 'focus-1', startSample: 100, endSample: 200, focusSource: 'vocals',
      confidence: 0.9, reason: 'vocal_presence', manuallyEdited: false,
    }];
    expect(focusSegmentAtSample(segments, 99)).toBeUndefined();
    expect(focusSegmentAtSample(segments, 100)?.id).toBe('focus-1');
    expect(focusSegmentAtSample(segments, 199)?.id).toBe('focus-1');
    expect(focusSegmentAtSample(segments, 200)).toBeUndefined();
  });

  it('falls back to a single mix lane when stems are absent or requested lanes are unavailable', () => {
    expect(resolveVisibleStemKinds([], ['vocals', 'drums'])).toEqual(['mix']);
    const descriptors: StemDescriptor[] = [
      { source: 'vocals', available: true, waveformUrl: '/vocals' },
      { source: 'drums', available: false, waveformUrl: '/drums' },
    ];
    expect(resolveVisibleStemKinds(descriptors, ['vocals', 'drums'])).toEqual(['vocals']);
    expect(resolveVisibleStemKinds(descriptors, [])).toEqual(['mix']);
  });
});

describe('stem and band filters', () => {
  beforeEach(() => resetEditorStore());

  it('combines semantic stem and spectral band without mutating hit data', () => {
    const points = [
      hit({ id: 'drum-low', band: 'low_hit', primaryStem: 'drums', stemEvidence: { drums: 0.95 } }),
      hit({ id: 'vocal-mid', band: 'mid_hit', primaryStem: 'vocals', stemEvidence: { vocals: 0.82 } }),
      hit({ id: 'drum-mid', band: 'mid_hit', primaryStem: 'drums', stemEvidence: { drums: 0.76 } }),
    ];
    const filters = {
      ...useEditorStore.getState().filters,
      band: 'mid_hit' as const,
      stem: 'drums' as const,
    };
    expect(filterHitPoints(points, filters).map((point) => point.id)).toEqual(['drum-mid']);
    expect(points).toHaveLength(3);
  });
});
