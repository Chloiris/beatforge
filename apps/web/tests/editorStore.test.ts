import { beforeEach, describe, expect, it } from 'vitest';
import { filterHitPoints, resetEditorStore, useEditorStore } from '../src/state/editorStore';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

function initialize() {
  useEditorStore.getState().initialize({
    trackId: 'track-1', signature: 'fixture', sampleRate, sampleCount,
    hitPoints: [hit(), hit({ id: 'hit-2', sample: 88_200, timeSec: 2, band: 'high_hit', confidence: 0.35 })],
    tempoMap: [tempo],
  });
}

describe('editor store', () => {
  beforeEach(() => { resetEditorStore(); initialize(); });

  it('adds and deletes an integer-sample hit point', () => {
    const point = useEditorStore.getState().addHitPoint(12_345.4);
    expect(point.sample).toBe(12_345);
    expect(useEditorStore.getState().hitPoints).toHaveLength(3);
    useEditorStore.getState().deleteSelected();
    expect(useEditorStore.getState().hitPoints).toHaveLength(2);
  });

  it('keeps the active marker stem available and visible as lanes change', () => {
    useEditorStore.getState().initialize({
      trackId: 'track-1',
      signature: 'fixture-with-stems',
      sampleRate,
      sampleCount,
      hitPoints: useEditorStore.getState().hitPoints,
      tempoMap: [tempo],
      availableStems: ['mix', 'vocals', 'drums'],
    });
    useEditorStore.getState().setActiveStem('vocals');
    expect(useEditorStore.getState().activeStem).toBe('vocals');

    useEditorStore.getState().setStemVisible('vocals', false);
    expect(useEditorStore.getState().activeStem).toBe('mix');

    useEditorStore.getState().setStemVisible('mix', false);
    expect(useEditorStore.getState()).toMatchObject({
      visibleStems: ['drums'],
      activeStem: 'drums',
    });

    useEditorStore.getState().initialize({
      trackId: 'track-1',
      signature: 'fixture-stems-normalized',
      sampleRate,
      sampleCount,
      hitPoints: useEditorStore.getState().hitPoints,
      tempoMap: [tempo],
      availableStems: ['mix', 'vocals'],
    });
    expect(useEditorStore.getState()).toMatchObject({
      availableStems: ['mix', 'vocals'],
      visibleStems: ['mix'],
      activeStem: 'mix',
    });
  });

  it('adds on the armed visible stem and relaxes filters that would hide the new point', () => {
    useEditorStore.getState().initialize({
      trackId: 'track-1',
      signature: 'fixture-manual-stem',
      sampleRate,
      sampleCount,
      hitPoints: useEditorStore.getState().hitPoints,
      tempoMap: [tempo],
      availableStems: ['mix', 'vocals'],
    });
    useEditorStore.getState().setActiveStem('vocals');
    useEditorStore.getState().updateFilters({
      stem: 'mix',
      band: 'high_hit',
      onlyUnedited: true,
      showHitPoints: false,
    });

    const point = useEditorStore.getState().addHitPoint(12_345.4);

    expect(point).toMatchObject({
      sample: 12_345,
      source: 'manual',
      primaryStem: 'vocals',
      stemEvidence: { vocals: 1 },
    });
    expect(useEditorStore.getState().filters).toMatchObject({
      stem: 'vocals',
      band: 'manual',
      onlyUnedited: false,
      showHitPoints: true,
    });
    expect(filterHitPoints(useEditorStore.getState().hitPoints, useEditorStore.getState().filters)).toContainEqual(point);
  });

  it('preserves analyzed stem evidence when the primary stem is manually reassigned', () => {
    useEditorStore.getState().initialize({
      trackId: 'track-1',
      signature: 'fixture-reassign-analysis-stem',
      sampleRate,
      sampleCount,
      hitPoints: [hit({ primaryStem: 'drums', stemEvidence: { drums: 0.92, bass: 0.3 } })],
      tempoMap: [tempo],
      availableStems: ['mix', 'vocals', 'drums', 'bass'],
    });

    useEditorStore.getState().updateHitPoint('hit-1', { primaryStem: 'vocals' });

    expect(useEditorStore.getState().hitPoints[0]).toMatchObject({
      source: 'fused',
      primaryStem: 'vocals',
      stemEvidence: { drums: 0.92, bass: 0.3, vocals: 0.92 },
      manuallyEdited: true,
    });
  });

  it('updates sample during a drag preview and commits one history entry', () => {
    const state = useEditorStore.getState();
    state.selectOnly('hit-1');
    state.beginPreview();
    state.moveHitPreview('hit-1', 50_000);
    state.moveHitPreview('hit-1', 51_000);
    expect(useEditorStore.getState().hitPoints[0].sample).toBe(51_000);
    state.commitPreview();
    expect(useEditorStore.getState().past).toHaveLength(1);
  });

  it('undoes and redoes edits', () => {
    useEditorStore.getState().selectOnly('hit-1');
    useEditorStore.getState().nudgeSelected(100);
    expect(useEditorStore.getState().hitPoints[0].sample).toBe(44_200);
    useEditorStore.getState().undo();
    expect(useEditorStore.getState().hitPoints[0].sample).toBe(44_100);
    useEditorStore.getState().redo();
    expect(useEditorStore.getState().hitPoints[0].sample).toBe(44_200);
  });

  it('filters by band, confidence and edit state without deleting data', () => {
    const points = useEditorStore.getState().hitPoints;
    const filters = { ...useEditorStore.getState().filters, band: 'high_hit' as const, minConfidence: 0.3 };
    expect(filterHitPoints(points, filters).map((point) => point.id)).toEqual(['hit-2']);
    expect(filterHitPoints(points, { ...filters, minConfidence: 0.5 })).toEqual([]);
    expect(useEditorStore.getState().hitPoints).toHaveLength(2);
  });
});
