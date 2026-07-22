import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChartDocument } from '../types';
import {
  clampPlaybackTime,
  DEFAULT_CHART_SCROLL_SPEED,
  type ChartScrollSpeed,
} from '../utils/chartPreview';
import { ChartTransport } from './ChartTransport';
import { FiveLaneChartPreview } from './FiveLaneChartPreview';

interface ChartPlayerProps {
  chart: ChartDocument;
  audioUrl: string;
  durationSec?: number;
}

export function ChartPlayer({ chart, audioUrl, durationSec = chart.durationSec }: ChartPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [mediaDurationSec, setMediaDurationSec] = useState(durationSec);
  const [playing, setPlaying] = useState(false);
  const [audioError, setAudioError] = useState('');
  const [scrollSpeed, setScrollSpeed] = useState<ChartScrollSpeed>(DEFAULT_CHART_SCROLL_SPEED);

  useEffect(() => {
    setPlaying(false);
    setCurrentTimeSec(0);
    setMediaDurationSec(durationSec);
    setAudioError('');
  }, [audioUrl, durationSec]);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const update = () => {
      const audio = audioRef.current;
      if (!audio) return;
      setCurrentTimeSec(audio.currentTime);
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      setAudioError('');
      void audio.play().catch((error: unknown) => {
        setPlaying(false);
        const detail = error instanceof Error && error.message ? `：${error.message}` : '';
        setAudioError(`无法播放参考音频${detail}`);
      });
    } else {
      audio.pause();
    }
  }, []);

  const seek = useCallback((nextTimeSec: number) => {
    const audio = audioRef.current;
    const next = clampPlaybackTime(nextTimeSec, mediaDurationSec);
    if (audio) audio.currentTime = next;
    setCurrentTimeSec(next);
  }, [mediaDurationSec]);

  const seekBy = useCallback((deltaSeconds: number) => {
    const sourceTime = audioRef.current?.currentTime ?? currentTimeSec;
    seek(sourceTime + deltaSeconds);
  }, [currentTimeSec, seek]);

  return (
    <section className="chart-player" aria-label="五轨谱面播放器">
      <audio
        ref={audioRef}
        src={audioUrl}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => {
          setPlaying(false);
          if (audioRef.current) setCurrentTimeSec(audioRef.current.currentTime);
        }}
        onEnded={() => {
          setPlaying(false);
          if (audioRef.current) setCurrentTimeSec(audioRef.current.duration || mediaDurationSec);
        }}
        onLoadedMetadata={() => {
          const measured = audioRef.current?.duration;
          if (measured && Number.isFinite(measured)) setMediaDurationSec(measured);
        }}
        onSeeked={() => {
          if (audioRef.current) setCurrentTimeSec(audioRef.current.currentTime);
        }}
        onError={() => {
          setPlaying(false);
          setAudioError('参考音频加载失败，请确认本地 SPEED 语料库可用。');
        }}
      />
      {audioError ? <div className="error-banner chart-audio-error" role="alert">{audioError}</div> : null}
      <FiveLaneChartPreview
        chart={chart}
        currentTimeSec={currentTimeSec}
        scrollSpeed={scrollSpeed}
      />
      <ChartTransport
        currentTimeSec={currentTimeSec}
        durationSec={mediaDurationSec}
        playing={playing}
        scrollSpeed={scrollSpeed}
        onTogglePlay={togglePlay}
        onSeek={seek}
        onSeekBy={seekBy}
        onScrollSpeedChange={setScrollSpeed}
      />
    </section>
  );
}
