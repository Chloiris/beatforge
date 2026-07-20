import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { MoraInspector } from '../src/components/MoraInspector';
import type {
  AlignmentHierarchy,
  AlignmentHierarchyUnit,
  CandidateEvent,
} from '../src/types';

const sampleRate = 44_100;

function hierarchyUnit(
  level: AlignmentHierarchyUnit['level'],
  index: number,
  text: string,
  phoneme: string | null = null,
): AlignmentHierarchyUnit {
  return {
    id: `${level}-${index}`,
    index,
    level,
    text,
    kana: level === 'mora' ? 'ホ' : null,
    mora: level === 'mora' ? 'ホ' : null,
    phoneme,
    kind: level,
    characterIndices: [0],
    moraIndices: [0],
    phonemeIndices: level === 'mora' ? [0, 1] : [index],
    alignedStartSample: 44_100,
    alignedEndSample: 45_100,
    refinedStartSample: 44_140,
    refinedEndSample: 45_140,
    alignedSample: 44_100,
    refinedSample: 44_140,
    confidence: 0.91,
    observedTokenIndex: index,
    matchOperation: 'match',
    evidence: level === 'mora'
      ? { energy: 0.5, spectralChange: 0.4, pitchChange: 0.3 }
      : null,
  };
}

const mora = hierarchyUnit('mora', 0, '星');
const hierarchy: AlignmentHierarchy = {
  characters: [hierarchyUnit('character', 0, '星')],
  moras: [mora],
  phonemes: [
    hierarchyUnit('phoneme', 0, '星', 'h'),
    hierarchyUnit('phoneme', 1, '星', 'o'),
  ],
};

function candidate(
  id: string,
  alignmentUnitIndex: number,
  hubert: number,
  overrides: Partial<CandidateEvent> = {},
): CandidateEvent {
  return {
    id,
    sample: 44_140,
    timeSec: 44_140 / sampleRate,
    acousticSample: 44_140,
    chartSample: 44_200,
    snapErrorMs: -60 / sampleRate * 1_000,
    lane: 'vocals',
    sourceEvidence: { vocals: 1 },
    semanticEvidence: {},
    confidence: 0.8,
    status: 'accepted',
    gridType: 'straight_1_16',
    gridConfidence: 0.64,
    source: 'vocals',
    generator: 'hubert_ctc',
    character: '星',
    mora: 'ホ',
    phoneme: 'h o',
    eventLevel: 'mora',
    eventPolicy: 'mora',
    alignmentUnitId: `mora-event:mora-${alignmentUnitIndex}`,
    alignmentUnitIndex,
    alignmentRunId: 'run-current',
    characterIndices: [0],
    phonemes: ['h', 'o'],
    alignedSample: 44_100,
    refinedSample: 44_140,
    evidence: {
      hubert,
      energy: 0.62,
      pitch: 0.63,
      rhythm: 0.64,
    },
    hitPointId: null,
    createdAt: '2026-07-19T00:00:00.000Z',
    updatedAt: '2026-07-19T00:00:00.000Z',
    ...overrides,
  };
}

describe('MoraInspector', () => {
  it('uses exact alignment identity or level/index and keeps Evidence collapsed by default', async () => {
    const user = userEvent.setup();
    const nearbyWrongUnit = candidate('nearby-wrong-unit', 8, 0.99);
    const exactLevelAndIndex = candidate('exact-level-index', 0, 0.61);

    render(
      <MoraInspector
        mora={mora}
        hierarchy={hierarchy}
        candidates={[nearbyWrongUnit, exactLevelAndIndex]}
        alignmentRunId="run-current"
        sampleRate={sampleRate}
        exportUrl="/api/tracks/track/export?format=json"
        onAlign={vi.fn()}
      />,
    );

    const inspector = screen.getByRole('complementary', { name: 'Mora Inspector' });
    expect(within(inspector).getByText('星')).toBeInTheDocument();
    expect(within(inspector).getByText('ホ')).toBeInTheDocument();
    expect(within(inspector).getByText('h o')).toBeInTheDocument();
    expect(within(inspector).getByText('exact-level-index')).toBeInTheDocument();
    expect(within(inspector).queryByText('nearby-wrong-unit')).not.toBeInTheDocument();

    const evidenceSummary = within(inspector).getByText('Evidence');
    const evidenceDetails = evidenceSummary.closest('details');
    expect(evidenceDetails).not.toHaveAttribute('open');
    await user.click(evidenceSummary);
    expect(evidenceDetails).toHaveAttribute('open');

    const evidence = within(inspector).getByLabelText('HuBERT Mora evidence');
    expect(within(evidence).getByText('61%')).toBeInTheDocument();
    expect(within(evidence).getByText('62%')).toBeInTheDocument();
    expect(within(evidence).getByText('63%')).toBeInTheDocument();
    expect(within(evidence).getByText('64%')).toBeInTheDocument();
    expect(within(evidence).queryByText('99%')).not.toBeInTheDocument();
  });

  it('rejects stale-run evidence even when the level and index match', () => {
    render(
      <MoraInspector
        mora={mora}
        hierarchy={hierarchy}
        candidates={[candidate('stale-candidate', 0, 0.98, { alignmentRunId: 'run-old' })]}
        alignmentRunId="run-current"
        sampleRate={sampleRate}
        exportUrl="/api/tracks/track/export?format=json"
        onAlign={vi.fn()}
      />,
    );

    expect(screen.getByText(/未找到精确关联的 CandidateEvent/)).toBeInTheDocument();
    expect(screen.queryByText('stale-candidate')).not.toBeInTheDocument();
    const evidence = screen.getByLabelText('HuBERT Mora evidence');
    expect(within(evidence).getAllByText('—')).toHaveLength(2);
    expect(within(evidence).queryByText('98%')).not.toBeInTheDocument();
  });

  it('seeks to the real refined sample, keeps destructive actions read-only, and exports', async () => {
    const user = userEvent.setup();
    const onAlign = vi.fn();
    render(
      <MoraInspector
        mora={mora}
        hierarchy={hierarchy}
        candidates={[candidate('exact-level-index', 0, 0.61)]}
        alignmentRunId="run-current"
        sampleRate={sampleRate}
        exportUrl="/api/tracks/track/export?format=json"
        onAlign={onAlign}
      />,
    );

    expect(screen.getByText('44,140 – 45,140')).toBeInTheDocument();
    expect(screen.getByText('01.001s')).toBeInTheDocument();
    expect(screen.queryByText('0:01.001 – 0:01.024')).not.toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '对齐' }));
    expect(onAlign).toHaveBeenCalledWith(44_140);
    expect(screen.getByRole('button', { name: '删除' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '锁定' })).toBeDisabled();
    expect(screen.getByRole('link', { name: '导出' })).toHaveAttribute(
      'href',
      '/api/tracks/track/export?format=json',
    );
    expect(screen.getByText(/HuBERT Mora 为只读声学证据/)).toBeInTheDocument();
  });
});
