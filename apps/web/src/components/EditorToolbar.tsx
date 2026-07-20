import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import type { AnalysisMode, ProjectDetail, TrackDetail } from '../types';
import { api } from '../api/client';
import { useEditorStore } from '../state/editorStore';
import { millisecondsToSamples } from '../utils/time';
import { Brand } from './Brand';

interface EditorToolbarProps {
  project: ProjectDetail;
  track: TrackDetail;
  mode: AnalysisMode;
  sensitivity: number;
  onModeChange: (mode: AnalysisMode) => void;
  onSensitivityChange: (sensitivity: number) => void;
  onAnalyze: () => void;
  onShowGuide: () => void;
  alignmentLabOpen: boolean;
  onToggleAlignmentLab: () => void;
  retrySave: () => void;
}

export function EditorToolbar({ project, track, mode, sensitivity, onModeChange, onSensitivityChange, onAnalyze, onShowGuide, alignmentLabOpen, onToggleAlignmentLab, retrySave }: EditorToolbarProps) {
  const tempo = useEditorStore((state) => state.tempoMap[0]);
  const subdivision = useEditorStore((state) => state.subdivision);
  const snapEnabled = useEditorStore((state) => state.snapEnabled);
  const pastCount = useEditorStore((state) => state.past.length);
  const futureCount = useEditorStore((state) => state.future.length);
  const saveStatus = useEditorStore((state) => state.saveStatus);
  const saveError = useEditorStore((state) => state.saveError);
  const revision = useEditorStore((state) => state.revision);
  const savedRevision = useEditorStore((state) => state.savedRevision);
  const updateTempo = useEditorStore((state) => state.updateTempo);
  const setSubdivision = useEditorStore((state) => state.setSubdivision);
  const setSnapEnabled = useEditorStore((state) => state.setSnapEnabled);
  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);
  const offsetMs = tempo ? (tempo.beatOffsetSample / track.originalSampleRate) * 1000 : 0;
  const rhythmConstraint = track.analysis?.rhythmConstraint;
  const currentHitCount = track.hitPoints.length;
  const vocalMoraSummary = useMemo(() => {
    const events = track.candidateEvents.filter((candidate) => (
      candidate.lane === 'vocals'
      && candidate.generator === 'hubert_ctc'
      && candidate.eventLevel === 'mora'
    ));
    const confidenceTotal = events.reduce(
      (total, event) => total + Math.max(0, Math.min(1, event.confidence)),
      0,
    );
    return {
      count: events.length,
      confidence: events.length ? confidenceTotal / events.length : null,
    };
  }, [track.candidateEvents]);
  const aiStatus = project.status === 'processing'
    ? {
        title: 'AI Analysis In Progress',
        detail: 'Generating vocal events',
        confidence: 'Confidence —',
      }
    : vocalMoraSummary.count
      ? {
          title: 'AI Analysis Complete',
          detail: `${vocalMoraSummary.count.toLocaleString()} vocal events generated`,
          confidence: `Confidence ${Math.round((vocalMoraSummary.confidence ?? 0) * 100)}%`,
        }
      : {
          title: 'AI Analysis Ready',
          detail: 'No Mora events generated',
          confidence: 'Confidence —',
        };
  const rhythmConstraintIsCurrent = Boolean(
    tempo
    && rhythmConstraint?.applied
    && Math.abs(rhythmConstraint.bpm - tempo.bpm) < 0.0001
    && rhythmConstraint.beatOffsetSample === tempo.beatOffsetSample,
  );

  return (
    <header className="editor-toolbar" aria-label="BeatForge Studio 编辑器">
      <div className="editor-topline">
        <div className="editor-header-left" role="group" aria-label="项目与歌曲">
          <Brand compact />
          <Link className="back-link" to="/">← 歌曲工作区</Link>
          <div className="editor-song-title">
            <strong>{project.title}</strong>
            <span>
              {project.artist || '未知艺术家'} · {track.originalFileName} · {' '}
              {track.originalSampleRate.toLocaleString()} Hz
            </span>
          </div>
        </div>

        <div
          className={`editor-work-status status-${project.status}`}
          role="status"
          aria-label="AI 工作状态"
          aria-live="polite"
        >
          <i aria-hidden="true" />
          <span><strong>{aiStatus.title}</strong><small>{aiStatus.detail}</small></span>
          <b>{aiStatus.confidence}</b>
        </div>

        <div className="editor-header-actions" role="group" aria-label="项目操作">
          <button
            className={`toolbar-button alignment-lab-button${alignmentLabOpen ? ' active' : ''}`}
            type="button"
            aria-pressed={alignmentLabOpen}
            aria-controls="vocal-alignment-panel"
            onClick={onToggleAlignmentLab}
          >Vocal Alignment</button>
          <button className="toolbar-button guide-button" onClick={onShowGuide}>使用引导</button>
          <button className="icon-button" onClick={undo} disabled={!pastCount} aria-label="撤销" title="撤销（Ctrl/⌘+Z）">↶</button>
          <button className="icon-button" onClick={redo} disabled={!futureCount} aria-label="重做" title="重做（Ctrl/⌘+Shift+Z）">↷</button>
          <div
            className={`save-indicator save-${saveStatus}`}
            role="status"
            aria-label="保存状态"
            aria-live="polite"
            title={saveError ?? ''}
          >
            <i />
            {saveStatus === 'saving' ? '正在保存' : saveStatus === 'error' ? '保存失败' : revision === savedRevision ? '已保存' : '等待保存'}
            {saveStatus === 'error' ? <button onClick={retrySave}>重试</button> : null}
          </div>
          <details className="export-menu">
            <summary className="primary-button">导出</summary>
            <div className="export-menu-panel">
              <span className="export-menu-heading">BeatForge 制谱包</span>
              <a
                className="export-option recommended"
                href={api.exportUrl(track.id, 'package', 'reference')}
                download
              >
                <strong>数据 + 参考音频</strong>
                <small>推荐 · 五轨标记与采样对齐 FLAC</small>
              </a>
              <a
                className="export-option"
                href={api.exportUrl(track.id, 'package', 'none')}
                download
              >
                <strong>仅数据包</strong>
                <small>五轨标记、双时间与分析信息</small>
              </a>
              <a
                className="export-option"
                href={api.exportUrl(track.id, 'package', 'full')}
                download
              >
                <strong>完整分轨包</strong>
                <small>参考音频 + Vocals / Drums / Bass / Other</small>
              </a>
              <span className="export-menu-heading legacy-heading">兼容格式</span>
              <div className="export-menu-legacy">
                <a href={api.exportUrl(track.id, 'json')} download>导出 JSON</a>
                <a href={api.exportUrl(track.id, 'csv')} download>导出 CSV</a>
              </div>
            </div>
          </details>
          <details className="toolbar-popover analysis-settings-menu">
            <summary className="toolbar-button">设置</summary>
            <div>
              <strong>重新生成候选点</strong>
              <label>分析模式<select aria-label="分析模式" value={mode} onChange={(event) => onModeChange(event.target.value as AnalysisMode)}><option value="recall">高召回</option><option value="balanced">平衡</option><option value="clean">干净</option><option value="accurate">精确</option></select></label>
              <label className="sensitivity-field">灵敏度 <input aria-label="灵敏度" type="range" min="0" max="1" step="0.05" value={sensitivity} onChange={(event) => onSensitivityChange(Number(event.target.value))} /><output>{Math.round(sensitivity * 100)}%</output></label>
              <button className="reanalyze-button" onClick={onAnalyze} type="button">↻ 重新分析整首歌曲</button>
              <small>普通编辑不需要重新分析。该操作会保留手工点和锁定点。</small>
            </div>
          </details>
        </div>
      </div>
      <div className="editor-controls-row">
        <span className="primary-controls-label">节奏网格</span>
        <label>BPM<input aria-label="BPM" className="numeric-input" type="number" min="20" max="400" step="0.01" value={tempo?.bpm ?? 120} onChange={(event) => updateTempo({ bpm: Math.max(20, Number(event.target.value)) })} /></label>
        {track.analysis && track.analysis.bpmConfidence < 0.5 ? <span className="tempo-warning" title="BPM 置信度较低">△ 建议检查</span> : null}
        <label>Offset<input aria-label="Offset milliseconds" className="numeric-input" type="number" step="0.1" value={Number(offsetMs.toFixed(3))} onChange={(event) => updateTempo({ beatOffsetSample: millisecondsToSamples(Number(event.target.value), track.originalSampleRate) })} /><small>ms</small></label>
        <label>细分<select aria-label="网格细分" value={subdivision} onChange={(event) => setSubdivision(event.target.value as typeof subdivision)}>{['1/4', '1/8', '1/12', '1/16', '1/24', '1/32'].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label className="switch-label">吸附 <button aria-label="吸附到网格" role="switch" aria-checked={snapEnabled} className={`toggle-switch${snapEnabled ? ' on' : ''}`} onClick={() => setSnapEnabled(!snapEnabled)}><i /></button></label>
        {rhythmConstraint?.applied ? <span className={`rhythm-constraint-badge${rhythmConstraintIsCurrent ? '' : ' stale'}`} title={rhythmConstraintIsCurrent ? `当前保存 ${currentHitCount} 个候选（包含已编辑点）；基础分析曾输出 ${rhythmConstraint.outputCount} 个 1/16 约束点，歌词对齐与静音校验可能替换或移除其中的点` : 'BPM 或 offset 已改变；现有自动点仍保留在上一次网格，请重新分析或手动吸附'}>{rhythmConstraintIsCurrent ? `当前候选 · ${currentHitCount} 点` : '△ 网格已变更'}</span> : null}
        <details className="toolbar-popover tempo-settings-menu">
          <summary>高级节拍⌄</summary>
          <div>
            <strong>精调网格</strong>
            <label>Offset sample<input aria-label="Offset sample" className="numeric-input sample-input" type="number" step="1" value={tempo?.beatOffsetSample ?? 0} onChange={(event) => updateTempo({ beatOffsetSample: Math.round(Number(event.target.value)) })} /><small>sample</small></label>
            <label>拍号<select value={tempo?.timeSignatureNumerator ?? 4} onChange={(event) => updateTempo({ timeSignatureNumerator: Number(event.target.value) })}><option value="3">3/4</option><option value="4">4/4</option><option value="5">5/4</option><option value="7">7/4</option></select></label>
          </div>
        </details>
      </div>
    </header>
  );
}
