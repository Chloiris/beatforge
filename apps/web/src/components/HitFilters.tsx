import type { CandidateLane, HitBand, StemKind } from '../types';
import { useEditorStore } from '../state/editorStore';
import { STEM_LABELS, STEM_ORDER } from '../utils/stems';

const bands: Array<{ value: 'all' | HitBand | 'manual'; label: string }> = [
  { value: 'all', label: '全部' }, { value: 'low_hit', label: '低频' }, { value: 'mid_hit', label: '中频' },
  { value: 'high_hit', label: '高频' }, { value: 'full_band_accent', label: '全频重音' }, { value: 'manual', label: '手动点' },
];
const candidateLanes: Array<{ value: 'all' | CandidateLane; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'vocals', label: 'Vocals' },
  { value: 'melody', label: 'Melody' },
  { value: 'drums', label: 'Drums' },
];

export function HitFilters() {
  const filters = useEditorStore((state) => state.filters);
  const updateFilters = useEditorStore((state) => state.updateFilters);
  const availableStems = useEditorStore((state) => state.availableStems);
  const visibleStems = useEditorStore((state) => state.visibleStems);
  const setStemVisible = useEditorStore((state) => state.setStemVisible);
  return (
    <div className="timeline-filter-stack">
      <div className="hit-filters">
        <div className="band-tabs" aria-label="击打点分类筛选">{bands.map((band) => <button key={band.value} className={filters.band === band.value ? 'active' : ''} onClick={() => updateFilters({ band: band.value })}><i className={`band-dot band-${band.value}`} />{band.label}</button>)}</div>
        <span className="filter-divider" />
        <div className="candidate-lane-tabs" aria-label="Candidate lane selector">
          <span>候选</span>
          {candidateLanes.map((lane) => (
            <button
              key={lane.value}
              type="button"
              className={filters.candidateLane === lane.value ? 'active' : ''}
              onClick={() => updateFilters({ candidateLane: lane.value })}
            >
              {lane.label}
            </button>
          ))}
          <i className="candidate-key accepted" title="accepted" />
          <i className="candidate-key uncertain" title="uncertain" />
          <i className="candidate-key alternative" title="alternative" />
        </div>
        <span className="filter-divider" />
        <label>置信度 ≥ <strong>{Math.round(filters.minConfidence * 100)}%</strong><input aria-label="置信度阈值" type="range" min="0" max="1" step="0.05" value={filters.minConfidence} onChange={(event) => updateFilters({ minConfidence: Number(event.target.value) })} /></label>
        <details className="display-options"><summary>显示选项⌄</summary><div>
          <label><input type="checkbox" checked={filters.onlyUnedited} onChange={(event) => updateFilters({ onlyUnedited: event.target.checked })} />只看未编辑</label>
          <label><input type="checkbox" checked={filters.onlyLowConfidence} onChange={(event) => updateFilters({ onlyLowConfidence: event.target.checked })} />只看低置信度</label>
          <label><input type="checkbox" checked={filters.onlyOffGrid} onChange={(event) => updateFilters({ onlyOffGrid: event.target.checked })} />只看离网格较远</label>
          <hr />
          <label><input type="checkbox" checked={filters.showGrid} onChange={(event) => updateFilters({ showGrid: event.target.checked })} />BPM 网格</label>
          <label><input type="checkbox" checked={filters.showHitPoints} onChange={(event) => updateFilters({ showHitPoints: event.target.checked })} />击打点</label>
          <label><input type="checkbox" checked={filters.showCandidateEvents} onChange={(event) => updateFilters({ showCandidateEvents: event.target.checked })} />候选事件</label>
          <label><input type="checkbox" checked={filters.showWaveform} onChange={(event) => updateFilters({ showWaveform: event.target.checked })} />波形</label>
        </div></details>
        <details className="stem-options">
          <summary>分轨与音源⌄</summary>
          <div className="stem-lane-controls" aria-label="分轨波形显示">
            <span className="stem-control-label">选择要查看的波形</span>
            <div className="stem-lane-buttons">
              {STEM_ORDER.map((source) => {
                const available = availableStems.includes(source);
                const visible = visibleStems.includes(source);
                return (
                  <button
                    key={source}
                    type="button"
                    aria-pressed={visible}
                    disabled={!available}
                    className={visible ? 'active' : ''}
                    onClick={() => setStemVisible(source, !visible)}
                    title={available ? `${visible ? '折叠' : '展开'}${STEM_LABELS[source]}波形` : `${STEM_LABELS[source]}分轨不可用`}
                  >
                    <i className={`stem-dot stem-${source}`} />{STEM_LABELS[source]}
                  </button>
                );
              })}
            </div>
            <label className="stem-hit-filter">只看这个音源的击打点
              <select
                aria-label="击打点主音源"
                value={filters.stem ?? 'all'}
                onChange={(event) => updateFilters({ stem: event.target.value as 'all' | StemKind })}
              >
                <option value="all">全部音源</option>
                {STEM_ORDER.filter((source) => availableStems.includes(source)).map((source) => <option key={source} value={source}>{STEM_LABELS[source]}</option>)}
              </select>
            </label>
            {availableStems.length === 1 ? <span className="mix-only-note">未生成分轨，当前仅显示 Mix</span> : null}
          </div>
        </details>
      </div>
    </div>
  );
}
