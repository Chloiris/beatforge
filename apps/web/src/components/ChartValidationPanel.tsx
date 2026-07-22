import type { ChartDocument } from '../types';

export function ChartValidationPanel({ chart }: { chart: ChartDocument | null }) {
  const validation = chart?.validation ?? null;
  const statistics = chart?.statistics ?? null;
  const fullStepReachable = validation?.metrics.fullStepReachable;
  return (
    <section className="chart-validation-panel" aria-label="谱面验证结果">
      <div className="chart-panel-heading">
        <span className="eyebrow">PLAYABILITY VALIDATOR</span>
        <h2>人体可玩性</h2>
      </div>
      {validation ? (
        <>
          <div className={`validator-score${validation.valid ? ' valid' : ' invalid'}`}>
            <strong>{validation.score.toFixed(0)}</strong>
            <span><b>{validation.valid ? '通过验证' : '需要修正'}</b><small>满分 100</small></span>
          </div>
          <div className="validator-metrics">
            <span><small>AVG NPS</small><strong>{statistics?.npsAverage.toFixed(2) ?? '—'}</strong></span>
            <span><small>PEAK NPS</small><strong>{statistics?.npsPeak.toFixed(2) ?? '—'}</strong></span>
            <span><small>JUMPS</small><strong>{statistics?.jumpCount ?? 0}</strong></span>
            <span><small>HOLDS</small><strong>{statistics?.holdCount ?? 0}</strong></span>
          </div>
          {typeof fullStepReachable === 'boolean' ? (
            <div className={`validator-footwork${fullStepReachable ? ' valid' : ' invalid'}`}>
              <span><small>FULL STEP · 全步伐</small><b>严格左右交替 · 无转圈</b></span>
              <strong>{fullStepReachable ? '通过' : '无解'}</strong>
            </div>
          ) : null}
          {validation.issues.length ? (
            <ol className="validator-issues">
              {validation.issues.map((issue, index) => (
                <li key={`${issue.code}:${issue.timeSec ?? index}`} className={`severity-${issue.severity}`}>
                  <span>{issue.severity}</span>
                  <strong>{issue.code.replaceAll('_', ' ')}</strong>
                  <p>{issue.message}</p>
                  {issue.timeSec !== null ? <small>{issue.timeSec.toFixed(3)} s · beat {issue.beat?.toFixed(3) ?? '—'}</small> : null}
                </li>
              ))}
            </ol>
          ) : <div className="validator-clean">没有发现密度、移动、同脚或非全步伐风险。</div>}
        </>
      ) : (
        <div className="validator-pending">生成谱面后显示 Validator 分数和问题定位。</div>
      )}
    </section>
  );
}
