import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { HitInspector } from '../src/components/HitInspector';
import { resetEditorStore, useEditorStore } from '../src/state/editorStore';
import type { TrackDetail } from '../src/types';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

const manualPoint = hit({
  id: 'manual-stem-hit',
  source: 'manual',
  primaryStem: 'drums',
  stemEvidence: { drums: 1 },
  detectorVotes: ['manual'],
  manuallyEdited: true,
});

const track: TrackDetail = {
  id: 'inspector-track',
  projectId: 'inspector-project',
  createdAt: '2026-07-21T00:00:00.000Z',
  updatedAt: '2026-07-21T00:00:00.000Z',
  originalFileName: 'inspector.wav',
  audioUrl: '/api/tracks/inspector-track/audio',
  format: 'wav',
  originalSampleRate: sampleRate,
  channels: 2,
  sampleCount,
  durationSec: sampleCount / sampleRate,
  leadingSilenceSamples: 0,
  analysis: null,
  tempoMap: [tempo],
  hitPoints: [manualPoint],
  candidateEvents: [],
  stems: [
    { source: 'mix', available: true, waveformUrl: '/waveform' },
    { source: 'vocals', available: true, waveformUrl: '/waveform-vocals' },
    { source: 'drums', available: true, waveformUrl: '/waveform-drums' },
  ],
  focusMap: [],
  waveformUrl: '/waveform',
};

describe('HitInspector stem reassignment', () => {
  beforeEach(() => {
    resetEditorStore();
    useEditorStore.getState().initialize({
      trackId: track.id,
      signature: 'inspector-stem-test',
      sampleRate,
      sampleCount,
      hitPoints: [manualPoint],
      tempoMap: [tempo],
      availableStems: ['mix', 'vocals', 'drums'],
    });
    useEditorStore.getState().setStemVisible('vocals', false);
    useEditorStore.getState().selectOnly(manualPoint.id);
  });

  it('reassigns the primary stem, evidence, visibility, and active marker target together', () => {
    render(<HitInspector track={track} />);

    const select = screen.getByLabelText('击打点主音源');
    expect(select).toHaveValue('drums');
    fireEvent.change(select, { target: { value: 'vocals' } });

    const state = useEditorStore.getState();
    expect(state.hitPoints[0]).toMatchObject({
      source: 'manual',
      primaryStem: 'vocals',
      stemEvidence: { vocals: 1 },
      manuallyEdited: true,
    });
    expect(state.hitPoints[0].stemEvidence).not.toHaveProperty('drums');
    expect(state.visibleStems).toContain('vocals');
    expect(state.activeStem).toBe('vocals');
  });
});
