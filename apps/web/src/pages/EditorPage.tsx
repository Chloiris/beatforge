import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { AnalysisProgress } from '../components/AnalysisProgress';
import { AlignmentLab } from '../components/AlignmentLab';
import { EditorToolbar } from '../components/EditorToolbar';
import { EditorQuickStart } from '../components/EditorQuickStart';
import { HitFilters } from '../components/HitFilters';
import { HitInspector } from '../components/HitInspector';
import { PlaybackControls } from '../components/PlaybackControls';
import { TimelineCanvas } from '../components/TimelineCanvas';
import { VocalLyricsPanel } from '../components/VocalLyricsPanel';
import { useAutoSave } from '../hooks/useAutoSave';
import { useEditorStore } from '../state/editorStore';
import type { AnalysisMode, StemKind } from '../types';
import { availableStemKinds } from '../utils/stems';
import { clampSample, millisecondsToSamples } from '../utils/time';

const EDITOR_GUIDE_STORAGE_KEY = 'beatforge:editor-guide:v3';

export function EditorPage() {
  const { projectId = '' } = useParams();
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [jobId, setJobId] = useState(searchParams.get('job'));
  const [mode, setMode] = useState<AnalysisMode>('balanced');
  const [sensitivity, setSensitivity] = useState(0.5);
  const [currentSample, setCurrentSample] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.85);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [follow, setFollow] = useState(true);
  const [loop, setLoop] = useState(false);
  const [actionError, setActionError] = useState('');
  const [audioError, setAudioError] = useState('');
  const [auditionSource, setAuditionSource] = useState<StemKind>('mix');
  const [guideOpen, setGuideOpen] = useState(
    () => globalThis.localStorage?.getItem(EDITOR_GUIDE_STORAGE_KEY) !== 'dismissed',
  );
  const [lyricsExpanded, setLyricsExpanded] = useState(false);
  const [alignmentLabOpen, setAlignmentLabOpen] = useState(false);
  const projectQuery = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
    enabled: Boolean(projectId),
    refetchInterval: (query) => query.state.data?.status === 'processing' && !jobId ? 1500 : false,
  });
  const track = projectQuery.data?.track ?? null;
  const waveformQuery = useQuery({
    queryKey: ['waveform', track?.id, 'mix', 'auto'],
    queryFn: () => api.getWaveform(track!.id, 'auto', 'mix'),
    enabled: Boolean(track?.id),
    staleTime: Infinity,
  });
  const jobQuery = useQuery({
    queryKey: ['analysis-job', jobId],
    queryFn: () => api.getJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'processing' || !status ? 700 : false;
    },
  });
  const retrySave = useAutoSave(track?.id);
  const auditionUrl = useMemo(() => {
    if (!track || auditionSource === 'mix') return track ? api.audioUrl(track.id) : '';
    return track.stems.find((stem) => stem.source === auditionSource && stem.available)?.audioUrl
      ?? api.audioUrl(track.id);
  }, [auditionSource, track]);

  useEffect(() => {
    if (!track) return;
    const stemSignature = (track.stems ?? []).filter((stem) => stem.available).map((stem) => stem.source).join(',');
    const signature = `${track.updatedAt}:${track.analysis?.createdAt ?? 'unprocessed'}:${track.hitPoints.length}:${track.candidateEvents?.length ?? 0}:${track.tempoMap[0]?.bpm ?? 0}:${stemSignature}:${track.focusMap?.length ?? 0}`;
    useEditorStore.getState().initialize({
      trackId: track.id,
      signature,
      sampleRate: track.originalSampleRate,
      sampleCount: track.sampleCount,
      hitPoints: track.hitPoints,
      tempoMap: track.tempoMap,
      availableStems: availableStemKinds(track.stems),
    });
    if (track.analysis?.mode) setMode(track.analysis.mode);
  }, [track]);

  useEffect(() => {
    if (!track || auditionSource === 'mix') return;
    if (!track.stems.some((stem) => stem.source === auditionSource && stem.available)) {
      setAuditionSource('mix');
    }
  }, [auditionSource, track]);

  useEffect(() => {
    if (jobQuery.data?.status === 'completed') {
      void queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      if (track?.id) void queryClient.invalidateQueries({ queryKey: ['waveform', track.id] });
    }
  }, [jobQuery.data?.status, projectId, queryClient, track?.id]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = volume;
  }, [volume]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.playbackRate = playbackRate;
    audio.preservesPitch = true;
  }, [playbackRate]);

  useEffect(() => {
    if (!playing || !track) return;
    let animationFrame = 0;
    const update = () => {
      const audio = audioRef.current;
      if (!audio) return;
      let sample = clampSample(Math.round(audio.currentTime * track.originalSampleRate), track.sampleCount);
      if (loop) {
        const state = useEditorStore.getState();
        const selection = state.hitPoints.filter((point) => state.selectedIds.includes(point.id)).map((point) => point.sample).sort((a, b) => a - b);
        if (selection.length >= 2 && sample >= selection.at(-1)!) {
          sample = selection[0];
          audio.currentTime = sample / track.originalSampleRate;
        }
      }
      setCurrentSample(sample);
      animationFrame = requestAnimationFrame(update);
    };
    animationFrame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animationFrame);
  }, [loop, playing, track]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      setAudioError('');
      void audio.play().catch((error: unknown) => {
        setPlaying(false);
        const detail = error instanceof Error && error.message ? `：${error.message}` : '';
        setAudioError(`无法播放音频${detail}`);
      });
    } else {
      audio.pause();
    }
  }, []);

  const stop = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause(); audio.currentTime = 0; setCurrentSample(0);
  }, []);

  const seek = useCallback((sample: number) => {
    if (!track || !audioRef.current) return;
    const safe = clampSample(sample, track.sampleCount);
    audioRef.current.currentTime = safe / track.originalSampleRate;
    setCurrentSample(safe);
  }, [track]);

  const analyze = useCallback(async () => {
    if (!track) return;
    setActionError('');
    try {
      const response = await api.analyzeTrack(track.id, mode, sensitivity);
      setJobId(response.jobId);
      await queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    } catch (error) {
      setActionError(error instanceof Error ? error.message : '无法启动分析任务');
    }
  }, [mode, projectId, queryClient, sensitivity, track]);

  useEffect(() => {
    if (!track) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(target.tagName) || target.isContentEditable)) return;
      const state = useEditorStore.getState();
      if (event.code === 'Space') { event.preventDefault(); togglePlay(); return; }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) state.redo(); else state.undo(); return; }
      if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); state.deleteSelected(); return; }
      if (event.key === 'Escape') { state.cancelPreview(); state.selectOnly(null); return; }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        const amount = event.altKey ? 1 : millisecondsToSamples(event.shiftKey ? 5 : 1, track.originalSampleRate);
        state.nudgeSelected(event.key === 'ArrowLeft' ? -amount : amount);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [togglePlay, track]);

  const project = projectQuery.data;
  const job = jobQuery.data;
  const mobileMessage = useMemo(() => <div className="mobile-readonly-note">移动端为浏览模式。建议使用桌面浏览器完成精确编辑。</div>, []);

  if (projectQuery.isLoading) return <div className="full-page-state"><span className="spinner large" /><strong>正在加载工程…</strong></div>;
  if (projectQuery.isError || !project) return <div className="full-page-state error-state"><strong>无法打开工程</strong><p>{projectQuery.error?.message ?? '工程不存在'}</p><Link to="/">返回歌曲工作区</Link></div>;
  if (!track) return <div className="full-page-state"><strong>工程还没有音轨</strong><Link to="/">返回歌曲工作区</Link></div>;

  const closeGuide = () => {
    globalThis.localStorage?.setItem(EDITOR_GUIDE_STORAGE_KEY, 'dismissed');
    setGuideOpen(false);
  };

  return (
    <main className="editor-shell">
      <EditorToolbar project={project} track={track} mode={mode} sensitivity={sensitivity} onModeChange={setMode} onSensitivityChange={setSensitivity} onAnalyze={() => void analyze()} onShowGuide={() => setGuideOpen(true)} alignmentLabOpen={alignmentLabOpen} onToggleAlignmentLab={() => setAlignmentLabOpen((open) => !open)} retrySave={retrySave} />
      <EditorQuickStart
        open={guideOpen && project.status !== 'processing'}
        onClose={closeGuide}
        onStartEditing={closeGuide}
        onStartLyrics={() => {
          closeGuide();
          setLyricsExpanded(true);
        }}
      />
      {mobileMessage}
      <audio
        ref={audioRef}
        src={auditionUrl}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => { setPlaying(false); setCurrentSample(track.sampleCount - 1); }}
        onLoadedMetadata={() => {
          setAudioError('');
          if (audioRef.current) {
            audioRef.current.currentTime = currentSample / track.originalSampleRate;
          }
        }}
        onError={() => {
          setPlaying(false);
          setAudioError('音频加载失败，请检查本地音频文件或重新加载。');
        }}
      />
      <div className={`editor-main${alignmentLabOpen ? ' alignment-lab-active' : ''}`}>
        <section className="timeline-column">
          {audioError ? <div className="error-banner editor-error">{audioError}<button onClick={() => { setAudioError(''); audioRef.current?.load(); }}>重新加载</button></div> : null}
          {actionError || jobQuery.isError ? <div className="error-banner editor-error">{actionError || jobQuery.error?.message}<button onClick={() => void analyze()}>重试</button></div> : null}
          {alignmentLabOpen ? (
            <AlignmentLab
              track={track}
              initialWaveform={waveformQuery.data}
              currentSample={currentSample}
              isPlaying={playing}
              followPlayback={follow}
              onSeek={seek}
              onClose={() => setAlignmentLabOpen(false)}
            />
          ) : (
            <>
              <HitFilters />
              <VocalLyricsPanel
                expanded={lyricsExpanded}
                onExpandedChange={setLyricsExpanded}
                onSeekSample={seek}
                trackId={track.id}
              />
              {job && job.status !== 'completed' ? <AnalysisProgress job={job} onRetry={job.status === 'failed' ? () => void analyze() : undefined} /> : null}
              {!job && project.status === 'processing' ? <div className="processing-banner"><span className="spinner" /> 分析任务正在后端运行，完成后将自动刷新。</div> : null}
              <div className="track-info-strip"><span><i className="band-dot band-low_hit" />LOW</span><span><i className="band-dot band-mid_hit" />MID</span><span><i className="band-dot band-high_hit" />HIGH</span><span><i className="band-dot band-full_band_accent" />ACCENT</span><div /><code>{track.format.toUpperCase()} · {track.channels} CH · {track.sampleCount.toLocaleString()} SAMPLES</code></div>
              <TimelineCanvas track={track} initialWaveform={waveformQuery.data} currentSample={currentSample} isPlaying={playing} followPlayback={follow} onSeek={seek} />
            </>
          )}
        </section>
        {alignmentLabOpen ? null : <HitInspector track={track} />}
      </div>
      <PlaybackControls track={track} currentSample={currentSample} playing={playing} volume={volume} playbackRate={playbackRate} follow={follow} loop={loop} auditionSource={auditionSource} onAuditionSource={(source) => { audioRef.current?.pause(); setAuditionSource(source); }} onTogglePlay={togglePlay} onStop={stop} onVolume={setVolume} onRate={setPlaybackRate} onFollow={setFollow} onLoop={setLoop} />
    </main>
  );
}
