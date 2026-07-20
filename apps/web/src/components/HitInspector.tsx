import { useMemo } from 'react';
import type { HitBand, StemKind, TrackDetail } from '../types';
import { useEditorStore } from '../state/editorStore';
import { primaryStemOf, STEM_COLORS, STEM_LABELS, STEM_ORDER } from '../utils/stems';
import { formatMusicalPosition, formatTime, sampleToSeconds, secondsToSample } from '../utils/time';

const bandLabels: Record<HitBand, string> = {
  low_hit: '低频击打', mid_hit: '中频击打', high_hit: '高频击打', full_band_accent: '全频重音',
};

export function HitInspector({ track }: { track: TrackDetail }) {
  const hitPoints = useEditorStore((state) => state.hitPoints);
  const selectedIds = useEditorStore((state) => state.selectedIds);
  const tempo = useEditorStore((state) => state.tempoMap[0]);
  const updateHitPoint = useEditorStore((state) => state.updateHitPoint);
  const availableStems = useEditorStore((state) => state.availableStems);
  const setStemVisible = useEditorStore((state) => state.setStemVisible);
  const setActiveStem = useEditorStore((state) => state.setActiveStem);
  const updateSelectedBand = useEditorStore((state) => state.updateSelectedBand);
  const setSelectedLocked = useEditorStore((state) => state.setSelectedLocked);
  const deleteSelected = useEditorStore((state) => state.deleteSelected);
  const snapSelected = useEditorStore((state) => state.snapSelected);
  const selected = useMemo(() => hitPoints.filter((point) => selectedIds.includes(point.id)), [hitPoints, selectedIds]);
  const point = selected.length === 1 ? selected[0] : null;

  if (!selected.length) {
    return (
      <aside className="inspector-panel">
        <header><span className="eyebrow">INSPECTOR</span><h2>属性检查器</h2></header>
        <div className="inspector-empty"><span>⌖</span><strong>先播放，再选择击打点</strong><p>按 Space 试听；单击竖线查看，拖动修正。Ctrl/⌘ 可多选，Ctrl/⌘+Z 可以撤销。</p></div>
        <div className="track-facts">
          <h3>音频基准</h3>
          <dl><div><dt>采样率</dt><dd>{track.originalSampleRate.toLocaleString()} Hz</dd></div><div><dt>总采样数</dt><dd>{track.sampleCount.toLocaleString()}</dd></div><div><dt>时长</dt><dd>{formatTime(track.durationSec)}</dd></div><div><dt>声道</dt><dd>{track.channels}</dd></div><div><dt>前导静音</dt><dd>{track.leadingSilenceSamples.toLocaleString()} smp</dd></div></dl>
        </div>
      </aside>
    );
  }

  if (!point) {
    const averageConfidence = selected.reduce((sum, item) => sum + item.confidence, 0) / selected.length;
    return (
      <aside className="inspector-panel">
        <header><span className="eyebrow">MULTI SELECTION</span><h2>已选 {selected.length} 个击打点</h2></header>
        <div className="multi-summary"><span>平均置信度</span><strong>{Math.round(averageConfidence * 100)}%</strong><i><b style={{ width: `${averageConfidence * 100}%` }} /></i></div>
        <label className="inspector-field">批量分类<select onChange={(event) => updateSelectedBand(event.target.value as HitBand)} defaultValue=""><option value="" disabled>选择类别…</option>{Object.entries(bandLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <div className="inspector-actions stacked"><button onClick={snapSelected}>⌁ 批量吸附到网格</button><button onClick={() => setSelectedLocked(true)}>◆ 批量锁定</button><button className="danger-button" onClick={deleteSelected}>删除选中点</button></div>
      </aside>
    );
  }

  const musical = tempo ? formatMusicalPosition(point.chartSample, track.originalSampleRate, tempo.bpm, tempo.beatOffsetSample, tempo.timeSignatureNumerator, 4) : '—';
  const primaryStem = primaryStemOf(point);
  const stemEvidence = STEM_ORDER
    .map((source) => ({ source, strength: point.stemEvidence?.[source] }))
    .filter((entry): entry is { source: typeof entry.source; strength: number } => typeof entry.strength === 'number')
    .sort((left, right) => right.strength - left.strength);
  const reassignPrimaryStem = (source: StemKind) => {
    setStemVisible(source, true);
    setActiveStem(source);
    updateHitPoint(point.id, { primaryStem: source });
  };
  return (
    <aside className="inspector-panel">
      <header><span className="eyebrow">HIT POINT</span><h2>击打点属性</h2><code>{point.id}</code></header>
      <div className={`inspector-band band-card-${point.band}`}><i /><div><small>粗分类</small><strong>{bandLabels[point.band]}</strong></div><span>{point.source}</span></div>
      <label className="inspector-stem-card">
        <span className={`stem-dot stem-${primaryStem}`} />
        <div>
          <small>主音源 / 标记轨道</small>
          <select
            aria-label="击打点主音源"
            value={primaryStem}
            onChange={(event) => reassignPrimaryStem(event.target.value as StemKind)}
          >
            {availableStems.map((source) => <option key={source} value={source}>{STEM_LABELS[source]}</option>)}
          </select>
        </div>
      </label>
      <label className="inspector-field">Acoustic sample<input aria-label="击打点 acoustic sample" type="number" step="1" value={point.acousticSample} onChange={(event) => updateHitPoint(point.id, { sample: Number(event.target.value) })} /></label>
      <label className="inspector-field">Acoustic time (sec)<input aria-label="击打点 acoustic time" type="number" step="0.001" value={Number(sampleToSeconds(point.acousticSample, track.originalSampleRate).toFixed(6))} onChange={(event) => updateHitPoint(point.id, { sample: secondsToSample(Number(event.target.value), track.originalSampleRate, track.sampleCount) })} /></label>
      <div className="position-readout"><span>{formatTime(point.timeSec)}</span><span>{musical}</span></div>
      <label className="inspector-field">Band<select value={point.band} onChange={(event) => updateHitPoint(point.id, { band: event.target.value as HitBand })}>{Object.entries(bandLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <div className="analysis-values">
        <ValueRow label="acousticSample" value={point.acousticSample.toLocaleString()} />
        <ValueRow label="chartSample" value={point.chartSample.toLocaleString()} />
        <ValueRow label="detectedSample" value={point.detectedSample.toLocaleString()} />
        <ValueRow label="refinedSample" value={point.refinedSample.toLocaleString()} />
        <ValueRow label="snappedSample" value={point.snappedSample.toLocaleString()} />
        <ValueRow label="snapErrorMs" value={`${point.snapErrorMs >= 0 ? '+' : ''}${point.snapErrorMs.toFixed(3)} ms`} warn={Math.abs(point.snapErrorMs) > 25} />
      </div>
      <div className="meter-field"><span>Confidence <strong>{point.confidence.toFixed(3)}</strong></span><i><b style={{ width: `${point.confidence * 100}%` }} /></i></div>
      <div className="meter-field salience-meter"><span>Salience <strong>{point.salience.toFixed(3)}</strong></span><i><b style={{ width: `${point.salience * 100}%` }} /></i></div>
      <div className="stem-evidence"><span>Stem evidence</span>{stemEvidence.length ? stemEvidence.map(({ source, strength }) => <div key={source}><label>{STEM_LABELS[source]}</label><i><b style={{ width: `${Math.max(0, Math.min(1, strength)) * 100}%`, background: STEM_COLORS[source] }} /></i><code>{Math.round(strength * 100)}%</code></div>) : <small>当前击打点没有逐分轨证据</small>}</div>
      <div className="detector-votes"><span>Detector votes</span><div>{point.detectorVotes.map((vote) => <code key={vote}>{vote}</code>)}</div></div>
      <dl className="boolean-facts"><div><dt>manuallyEdited</dt><dd>{String(point.manuallyEdited)}</dd></div><div><dt>locked</dt><dd>{String(point.locked)}</dd></div></dl>
      <div className="inspector-actions"><button onClick={snapSelected}>⌁ 吸附</button><button onClick={() => updateHitPoint(point.id, { sample: point.detectedSample })}>回到检测点</button><button onClick={() => updateHitPoint(point.id, { sample: point.refinedSample })}>回到精修点</button></div>
      <label className="lock-row"><span>锁定击打点<small>锁定后不可拖动或删除</small></span><button role="switch" aria-checked={point.locked} className={`toggle-switch${point.locked ? ' on' : ''}`} onClick={() => setSelectedLocked(!point.locked)}><i /></button></label>
      <button className="danger-button full-button" disabled={point.locked} onClick={deleteSelected}>删除击打点</button>
    </aside>
  );
}

function ValueRow({ label, value, warn = false }: { label: string; value: string; warn?: boolean }) {
  return <div><span>{label}</span><code className={warn ? 'warning-value' : ''}>{value}</code></div>;
}
