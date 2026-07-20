import type { AnalysisJob } from '../types';

const stages = [
  ['upload', '上传文件'],
  ['decode', '解码音频'],
  ['waveform', '提取波形'],
  ['bpm', '估计 BPM'],
  ['features', '计算多频段特征'],
  ['detect', '检测候选点'],
  ['refine', '局部时间精修'],
  ['merge', '合并和分类'],
  ['save', '保存结果'],
] as const;

const aliases: Record<string, string> = {
  queued: 'upload', uploading: 'upload', decoding: 'decode', decoded: 'decode',
  decoding_audio: 'decode', extracting_waveform: 'waveform', waveform: 'waveform', estimating_bpm: 'bpm', tempo: 'bpm',
  extracting_features: 'features', feature_extraction: 'features', computing_multiband_features: 'features', source_separation: 'features',
  detecting_onsets: 'detect', candidate_detection: 'detect', detecting_and_refining_candidates: 'refine',
  refining: 'refine', time_refinement: 'refine', merging: 'merge', classification: 'merge',
  merging_classifying_and_serializing: 'merge', saving: 'save', saving_results: 'save', completed: 'save', analysis_complete: 'save',
};

export function AnalysisProgress({ job, onRetry }: { job: AnalysisJob; onRetry?: () => void }) {
  const stageKey = aliases[job.stage] ?? job.stage;
  const currentIndex = Math.max(0, stages.findIndex(([key]) => key === stageKey));
  const completedKeys = new Set(Object.keys(job.stageTimings).map((key) => aliases[key] ?? key));
  const normalizedProgress = job.progress > 1 ? job.progress : job.progress * 100;
  return (
    <aside className={`analysis-progress analysis-${job.status}`} aria-live="polite">
      <div className="analysis-progress-head">
        <div><span className="eyebrow">LOCAL AUDIO PIPELINE</span><strong>{job.status === 'failed' ? '分析失败' : job.status === 'completed' ? '分析完成' : '正在生成击打点'}</strong></div>
        <span className="analysis-percent">{Math.round(normalizedProgress)}%</span>
      </div>
      <div className="analysis-meter"><i style={{ width: `${Math.max(0, Math.min(100, normalizedProgress))}%` }} /></div>
      <ol className="stage-list">
        {stages.map(([key, label], index) => {
          const timing = job.stageTimings[key];
          const state = job.status === 'completed' || completedKeys.has(key) ? 'done' : key === stageKey && job.status !== 'failed' ? 'active' : index < currentIndex ? 'done' : 'pending';
          return <li key={key} className={state}><i>{state === 'done' ? '✓' : index + 1}</i><span>{label}</span>{timing !== undefined ? <small>{timing.toFixed(0)} ms</small> : null}</li>;
        })}
      </ol>
      {job.warnings.length ? <div className="analysis-warning">{job.warnings.join('；')}</div> : null}
      {job.error ? <div className="error-banner">{typeof job.error === 'string' ? job.error : job.error.message}{onRetry ? <button onClick={onRetry}>重试</button> : null}</div> : null}
    </aside>
  );
}
