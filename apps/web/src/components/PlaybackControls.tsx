import type { StemKind, TrackDetail } from '../types';
import { availableStemKinds, STEM_LABELS } from '../utils/stems';
import { formatTime, sampleToSeconds } from '../utils/time';

interface PlaybackControlsProps {
  track: TrackDetail;
  currentSample: number;
  playing: boolean;
  volume: number;
  playbackRate: number;
  follow: boolean;
  loop: boolean;
  auditionSource: StemKind;
  onAuditionSource: (source: StemKind) => void;
  onTogglePlay: () => void;
  onStop: () => void;
  onVolume: (volume: number) => void;
  onRate: (rate: number) => void;
  onFollow: (follow: boolean) => void;
  onLoop: (loop: boolean) => void;
}

export function PlaybackControls(props: PlaybackControlsProps) {
  const { track, currentSample, playing, volume, playbackRate, follow, loop } = props;
  return (
    <footer className="playback-controls">
      <button className="transport-secondary" aria-label="回到开头" onClick={props.onStop}>■</button>
      <button className="play-button" aria-label={playing ? '暂停' : '播放'} onClick={props.onTogglePlay}>{playing ? 'Ⅱ' : '▶'}</button>
      <div className="transport-time"><strong>{formatTime(sampleToSeconds(currentSample, track.originalSampleRate))}</strong><span>/ {formatTime(track.durationSec)}</span></div>
      <div className="transport-sample"><small>CURRENT SAMPLE</small><code>{currentSample.toLocaleString()}</code></div>
      <div className="transport-spacer" />
      <label className="stem-audition-control">试听<select aria-label="试听音源" value={props.auditionSource} onChange={(event) => props.onAuditionSource(event.target.value as StemKind)}>{availableStemKinds(track.stems).map((source) => <option key={source} value={source}>{STEM_LABELS[source]}</option>)}</select></label>
      <label className="transport-toggle"><input type="checkbox" checked={loop} onChange={(event) => props.onLoop(event.target.checked)} />↻ 循环选区</label>
      <label className="transport-toggle"><input type="checkbox" checked={follow} onChange={(event) => props.onFollow(event.target.checked)} />播放时跟随</label>
      <label className="rate-control">速度<select aria-label="播放速度" value={playbackRate} onChange={(event) => props.onRate(Number(event.target.value))}><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1">1.0×</option></select></label>
      <label className="volume-control"><span>{volume === 0 ? '♩' : '♪'}</span><input aria-label="音量" type="range" min="0" max="1" step="0.01" value={volume} onChange={(event) => props.onVolume(Number(event.target.value))} /></label>
    </footer>
  );
}
