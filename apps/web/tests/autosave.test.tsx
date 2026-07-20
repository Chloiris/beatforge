import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '../src/api/client';
import { useAutoSave } from '../src/hooks/useAutoSave';
import { resetEditorStore, useEditorStore } from '../src/state/editorStore';
import { hit, sampleCount, sampleRate, tempo } from './fixtures';

describe('autosave', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    resetEditorStore();
    useEditorStore.getState().initialize({ trackId: 'track-1', signature: 'fixture', sampleRate, sampleCount, hitPoints: [hit()], tempoMap: [tempo] });
  });

  it('debounces edits and exposes saved status', async () => {
    const saveHits = vi.spyOn(api, 'saveHitPoints').mockResolvedValue([]);
    const saveTempo = vi.spyOn(api, 'saveTempoMap').mockResolvedValue([]);
    renderHook(() => useAutoSave('track-1', 300));
    act(() => { useEditorStore.getState().selectOnly('hit-1'); useEditorStore.getState().nudgeSelected(44); });
    expect(useEditorStore.getState().saveStatus).toBe('idle');
    await act(async () => { await vi.advanceTimersByTimeAsync(299); });
    expect(saveHits).not.toHaveBeenCalled();
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(saveHits).toHaveBeenCalledTimes(1);
    expect(saveTempo).toHaveBeenCalledTimes(1);
    expect(useEditorStore.getState().saveStatus).toBe('saved');
    expect(useEditorStore.getState().savedRevision).toBe(useEditorStore.getState().revision);
  });
});
