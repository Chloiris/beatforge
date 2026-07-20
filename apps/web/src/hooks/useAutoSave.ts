import { useCallback, useEffect, useRef } from 'react';
import { api } from '../api/client';
import { useEditorStore } from '../state/editorStore';
import { sampleToSeconds } from '../utils/time';

export function useAutoSave(trackId: string | undefined, debounceMs = 700): () => void {
  const revision = useEditorStore((state) => state.revision);
  const savedRevision = useEditorStore((state) => state.savedRevision);
  const retryToken = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const save = useCallback(async () => {
    const state = useEditorStore.getState();
    if (!trackId || state.revision === state.savedRevision) return;
    const savingRevision = state.revision;
    state.setSaveStatus('saving');
    const hitPoints = state.hitPoints.map((point) => ({
      ...point,
      sample: Math.round(point.sample),
      timeSec: sampleToSeconds(point.sample, state.sampleRate),
    }));
    try {
      // Keep the two persistent views ordered: tempo save recalculates server-side snap
      // recommendations, then the sample-truth hit payload writes the matching values.
      await api.saveTempoMap(trackId, state.tempoMap);
      await api.saveHitPoints(trackId, hitPoints);
      useEditorStore.getState().markSaved(savingRevision);
    } catch (error) {
      useEditorStore
        .getState()
        .setSaveStatus('error', error instanceof Error ? error.message : '保存失败');
    }
  }, [trackId]);

  const retry = useCallback(() => {
    retryToken.current += 1;
    if (timerRef.current) clearTimeout(timerRef.current);
    void save();
  }, [save]);

  useEffect(() => {
    if (!trackId || revision === savedRevision) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => void save(), debounceMs);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [debounceMs, revision, savedRevision, save, trackId]);

  return retry;
}
