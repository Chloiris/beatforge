import { formatTime } from '../utils/time';
import {
  CHART_SCROLL_SPEED_OPTIONS,
  isChartScrollSpeed,
  type ChartScrollSpeed,
} from '../utils/chartPreview';

interface ChartTransportProps {
  currentTimeSec: number;
  durationSec: number;
  playing: boolean;
  scrollSpeed: ChartScrollSpeed;
  disabled?: boolean;
  onTogglePlay: () => void;
  onSeek: (timeSec: number) => void;
  onSeekBy: (deltaSeconds: number) => void;
  onScrollSpeedChange: (scrollSpeed: ChartScrollSpeed) => void;
}

export function ChartTransport({
  currentTimeSec,
  durationSec,
  playing,
  scrollSpeed,
  disabled = false,
  onTogglePlay,
  onSeek,
  onSeekBy,
  onScrollSpeedChange,
}: ChartTransportProps) {
  const safeDuration = Math.max(0, durationSec);
  const safeCurrent = Math.max(0, Math.min(safeDuration, currentTimeSec));
  return (
    <div className="chart-transport" aria-label="谱面预览播放控制">
      <button
        className="transport-secondary"
        type="button"
        aria-label="快退 5 秒"
        disabled={disabled}
        onClick={() => onSeekBy(-5)}
      >−5</button>
      <button
        className="play-button"
        type="button"
        aria-label={playing ? '暂停谱面预览' : '播放谱面预览'}
        disabled={disabled}
        onClick={onTogglePlay}
      >{playing ? 'Ⅱ' : '▶'}</button>
      <button
        className="transport-secondary"
        type="button"
        aria-label="快进 5 秒"
        disabled={disabled}
        onClick={() => onSeekBy(5)}
      >+5</button>
      <div className="chart-transport-time">
        <strong>{formatTime(safeCurrent)}</strong>
        <span>/ {formatTime(safeDuration)}</span>
      </div>
      <label className="chart-scroll-speed-control">
        <span>流速</span>
        <select
          aria-label="谱面流速"
          value={scrollSpeed}
          disabled={disabled}
          onChange={(event) => {
            const nextSpeed = Number(event.target.value);
            if (isChartScrollSpeed(nextSpeed)) onScrollSpeedChange(nextSpeed);
          }}
        >
          {CHART_SCROLL_SPEED_OPTIONS.map((speed) => (
            <option key={speed} value={speed}>{speed}×</option>
          ))}
        </select>
      </label>
      <label className="chart-seek-control">
        <span className="sr-only">播放位置</span>
        <input
          aria-label="播放位置"
          type="range"
          min="0"
          max={Math.max(0.001, safeDuration)}
          step="0.001"
          value={safeCurrent}
          disabled={disabled}
          onChange={(event) => onSeek(Number(event.target.value))}
        />
      </label>
    </div>
  );
}
