import type {
  AlignmentHierarchy,
  AlignmentHierarchyUnit,
  AlignmentReport,
  CandidateEvent,
} from '../types';
import { sampleToSeconds } from '../utils/time';

interface MoraInspectorProps {
  mora: AlignmentHierarchyUnit | null;
  hierarchy: AlignmentHierarchy | null | undefined;
  candidates: CandidateEvent[];
  alignmentRunId: string | null;
  sampleRate: number;
  exportUrl: string;
  onAlign: (sample: number) => void;
  report?: AlignmentReport;
  reportLoading?: boolean;
  reportError?: string;
}

function percent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function refinedTime(sample: number, sampleRate: number): string {
  const [seconds, milliseconds] = sampleToSeconds(sample, sampleRate).toFixed(3).split('.');
  return `${seconds.padStart(2, '0')}.${milliseconds}s`;
}

function candidatesForRun(
  candidates: CandidateEvent[],
  alignmentRunId: string | null,
): CandidateEvent[] {
  return candidates.filter((candidate) => (
    candidate.generator === 'hubert_ctc'
    && (!alignmentRunId || candidate.alignmentRunId === alignmentRunId)
  ));
}

/**
 * Resolve evidence only through stable alignment identity. Sample proximity is
 * intentionally excluded: nearby Mora events are common in fast Japanese vocals.
 */
function findCandidateForAlignmentUnit(
  unit: AlignmentHierarchyUnit,
  candidates: CandidateEvent[],
  alignmentRunId: string | null,
): CandidateEvent | null {
  const eligible = candidatesForRun(candidates, alignmentRunId);
  const identityMatch = eligible.find(
    (candidate) => candidate.alignmentUnitId === unit.id,
  );
  if (identityMatch) return identityMatch;

  return eligible.find((candidate) => (
    candidate.eventLevel === unit.level
    && candidate.alignmentUnitIndex === unit.index
  )) ?? null;
}

function relatedLabel(
  indices: number[],
  units: AlignmentHierarchyUnit[] | undefined,
  label: (unit: AlignmentHierarchyUnit) => string | null,
): string {
  if (!units?.length) return '—';
  const byIndex = new Map(units.map((unit) => [unit.index, unit]));
  const values = indices.flatMap((index) => {
    const unit = byIndex.get(index);
    const value = unit ? label(unit) : null;
    return value ? [value] : [];
  });
  return values.length ? values.join(' ') : '—';
}

function EvidenceMeter({ label, value }: { label: string; value: number | null | undefined }) {
  const normalized = value === null || value === undefined || !Number.isFinite(value)
    ? null
    : Math.max(0, Math.min(1, value));
  return (
    <div className="alignment-score-metric mora-evidence-meter">
      <span>{label}<strong>{percent(normalized)}</strong></span>
      <i><b style={{ width: normalized === null ? '0%' : `${normalized * 100}%` }} /></i>
    </div>
  );
}

export function MoraInspector({
  mora,
  hierarchy,
  candidates,
  alignmentRunId,
  sampleRate,
  exportUrl,
  onAlign,
  report,
  reportLoading = false,
  reportError = '',
}: MoraInspectorProps) {
  const candidate = mora
    ? findCandidateForAlignmentUnit(mora, candidates, alignmentRunId)
    : null;
  const character = mora
    ? relatedLabel(mora.characterIndices, hierarchy?.characters, (unit) => unit.text)
    : '—';
  const phonemesFromHierarchy = mora
    ? relatedLabel(
      mora.phonemeIndices,
      hierarchy?.phonemes,
      (unit) => unit.phoneme || unit.text,
    )
    : '—';
  const phonemes = phonemesFromHierarchy !== '—'
    ? phonemesFromHierarchy
    : candidate?.phonemes?.join(' ') || mora?.phoneme || '—';
  const moraLabel = mora?.mora || mora?.kana || mora?.text || '—';
  const hubertEvidence = candidate?.evidence?.hubert;
  const energyEvidence = candidate?.evidence?.energy ?? mora?.evidence?.energy;
  const pitchEvidence = candidate?.evidence?.pitch ?? mora?.evidence?.pitchChange;
  const rhythmEvidence = candidate?.evidence?.rhythm;

  return (
    <aside className="alignment-score-panel mora-inspector" aria-label="Mora Inspector">
      <header>
        <span className="eyebrow">MORA INSPECTOR</span>
        <h3>发音事件</h3>
      </header>

      {mora ? (
        <section className="mora-inspector-detail" data-mora-id={mora.id}>
          <div className="mora-inspector-identity">
            <div><span>Character</span><strong>{character}</strong></div>
            <div><span>Mora</span><strong>{moraLabel}</strong></div>
            <div><span>Phoneme</span><strong>{phonemes}</strong></div>
          </div>

          <dl className="mora-inspector-facts">
            <div>
              <dt>Sample</dt>
              <dd>{mora.refinedStartSample.toLocaleString()} – {mora.refinedEndSample.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Time</dt>
              <dd>{refinedTime(mora.refinedSample, sampleRate)}</dd>
            </div>
            <div><dt>Anchor</dt><dd>{mora.refinedSample.toLocaleString()} sample</dd></div>
            <div><dt>Confidence</dt><dd>{percent(mora.confidence)}</dd></div>
          </dl>

          <details className="mora-inspector-evidence">
            <summary>Evidence</summary>
            <div aria-label="HuBERT Mora evidence">
              <EvidenceMeter label="HuBERT" value={hubertEvidence} />
              <EvidenceMeter label="Energy" value={energyEvidence} />
              <EvidenceMeter label="Pitch" value={pitchEvidence} />
              <EvidenceMeter label="Rhythm" value={rhythmEvidence} />
            </div>
          </details>

          <div className="mora-candidate-link">
            {candidate ? (
              <><span>CandidateEvent</span><code>{candidate.id}</code></>
            ) : (
              <span>未找到精确关联的 CandidateEvent；未使用附近采样点推测。</span>
            )}
          </div>

          <div className="mora-inspector-actions" aria-label="Mora 操作">
            <button type="button" onClick={() => onAlign(mora.refinedSample)}>对齐</button>
            <button type="button" disabled title="HuBERT Mora 是只读分析结果；请在编辑器中删除击打点。">删除</button>
            <button type="button" disabled title="HuBERT Mora 是只读分析结果；锁定仅适用于编辑器击打点。">锁定</button>
            <a className="toolbar-button" href={exportUrl} download>导出</a>
          </div>
          <p className="mora-inspector-readonly">
            HuBERT Mora 为只读声学证据。删除与锁定只操作编辑器中的击打点。
          </p>
        </section>
      ) : (
        <div className="alignment-score-empty mora-inspector-empty">
          完成 HuBERT 对齐后，这里会显示首个真实 Mora。
        </div>
      )}

      <section className="mora-proxy-summary" aria-label="Proxy Evaluation">
        <header><span className="eyebrow">PROXY EVALUATION</span><h3>HuBERT Score</h3></header>
        {!report ? (
          <div className="alignment-score-empty">
            {reportLoading ? <><span className="spinner" /> 正在读取评分…</> : '完成 HuBERT 对齐后显示评分。'}
          </div>
        ) : (
          <div className="alignment-score-detail">
            <div className="alignment-score-total"><span>Japanese HuBERT CTC</span><strong>{percent(report.score)}</strong></div>
            <EvidenceMeter label="Coverage" value={report.coverage} />
            <EvidenceMeter label="Acoustic" value={report.acoustic} />
            <EvidenceMeter label="Rhythm" value={report.rhythm} />
            <EvidenceMeter label="Stability" value={report.stability} />
            <dl>
              <div><dt>Lyric tokens</dt><dd>{report.lyricTokenCount.toLocaleString()}</dd></div>
              <div><dt>Aligned tokens</dt><dd>{report.alignedTokenCount.toLocaleString()}</dd></div>
            </dl>
          </div>
        )}
        {reportError ? <div className="alignment-score-error">评分读取失败：{reportError}</div> : null}
      </section>
    </aside>
  );
}
