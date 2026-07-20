import { useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type {
  AlignmentHierarchy,
  AlignmentHierarchyUnit,
  AlignmentLayer,
} from '../types';

interface VocalLyricsHierarchyProps {
  hierarchy: AlignmentHierarchy;
  onSeekSample?: (sample: number) => void;
}

function relatedUnits(
  indices: number[],
  unitsByIndex: Map<number, AlignmentHierarchyUnit>,
  level: AlignmentLayer,
): AlignmentHierarchyUnit[] {
  return indices.flatMap((index) => {
    const unit = unitsByIndex.get(index);
    return unit?.level === level ? [unit] : [];
  });
}

function moraLabel(unit: AlignmentHierarchyUnit): string {
  return unit.mora || unit.kana || unit.text || '·';
}

function phonemeLabel(unit: AlignmentHierarchyUnit): string {
  return unit.phoneme || unit.text || '·';
}

function refinedRange(unit: AlignmentHierarchyUnit): string {
  return `${unit.refinedStartSample.toLocaleString()}–${unit.refinedEndSample.toLocaleString()}`;
}

export function VocalLyricsHierarchy({
  hierarchy,
  onSeekSample,
}: VocalLyricsHierarchyProps) {
  const [expandedCharacters, setExpandedCharacters] = useState<Set<string>>(() => new Set());
  const [expandedMoras, setExpandedMoras] = useState<Set<string>>(() => new Set());
  const morasByIndex = useMemo(
    () => new Map(hierarchy.moras.map((unit) => [unit.index, unit])),
    [hierarchy.moras],
  );
  const phonemesByIndex = useMemo(
    () => new Map(hierarchy.phonemes.map((unit) => [unit.index, unit])),
    [hierarchy.phonemes],
  );

  const toggle = (
    id: string,
    setter: Dispatch<SetStateAction<Set<string>>>,
  ) => setter((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });

  return (
    <section className="lyrics-hierarchy" aria-label="歌词层级 Character → Mora → Phoneme">
      <header>
        <div><span className="eyebrow">HUBERT LYRICS</span><strong>歌词发音层级</strong></div>
        <code>
          {hierarchy.characters.length.toLocaleString()} Character · {' '}
          {hierarchy.moras.length.toLocaleString()} Mora · {' '}
          {hierarchy.phonemes.length.toLocaleString()} Phoneme
        </code>
      </header>
      <ol className="lyrics-character-list">
        {hierarchy.characters.map((character) => {
          const moras = relatedUnits(character.moraIndices, morasByIndex, 'mora');
          const characterExpanded = expandedCharacters.has(character.id);
          return (
            <li key={character.id} data-unit-level="character" data-unit-index={character.index}>
              <div className="lyrics-hierarchy-row character-row">
                <button
                  type="button"
                  aria-expanded={characterExpanded}
                  aria-label={`${characterExpanded ? '收起' : '展开'} Character ${character.index + 1} ${character.text || '·'}`}
                  onClick={() => toggle(character.id, setExpandedCharacters)}
                >
                  <code>{String(character.index + 1).padStart(3, '0')}</code>
                  <strong>{character.text || '·'}</strong>
                  <span>{character.kana || '—'} · {moras.length} Mora</span>
                  <small>{refinedRange(character)}</small>
                  <b aria-hidden="true">{characterExpanded ? '−' : '+'}</b>
                </button>
                {onSeekSample ? (
                  <button
                    className="lyrics-hierarchy-seek"
                    type="button"
                    aria-label={`试听 Character ${character.index + 1} ${character.text || '·'}`}
                    onClick={() => onSeekSample(character.refinedSample)}
                  >▶</button>
                ) : null}
              </div>
              {characterExpanded ? (
                <ol className="lyrics-mora-list" aria-label={`Character ${character.index + 1} 的 Mora`}>
                  {moras.map((mora) => {
                    const phonemes = relatedUnits(mora.phonemeIndices, phonemesByIndex, 'phoneme');
                    const moraExpanded = expandedMoras.has(mora.id);
                    const label = moraLabel(mora);
                    return (
                      <li key={mora.id} data-unit-level="mora" data-unit-index={mora.index}>
                        <div className="lyrics-hierarchy-row mora-row">
                          <button
                            type="button"
                            aria-expanded={moraExpanded}
                            aria-label={`${moraExpanded ? '收起' : '展开'} Mora ${mora.index + 1} ${label}`}
                            onClick={() => toggle(mora.id, setExpandedMoras)}
                          >
                            <code>M{String(mora.index + 1).padStart(3, '0')}</code>
                            <strong>{label}</strong>
                            <span>{phonemes.length} Phoneme</span>
                            <small>{refinedRange(mora)}</small>
                            <b aria-hidden="true">{moraExpanded ? '−' : '+'}</b>
                          </button>
                          {onSeekSample ? (
                            <button
                              className="lyrics-hierarchy-seek"
                              type="button"
                              aria-label={`试听 Mora ${mora.index + 1} ${label}`}
                              onClick={() => onSeekSample(mora.refinedSample)}
                            >▶</button>
                          ) : null}
                        </div>
                        {moraExpanded ? (
                          <ol className="lyrics-phoneme-list" aria-label={`Mora ${mora.index + 1} 的 Phoneme`}>
                            {phonemes.map((phoneme) => {
                              const phone = phonemeLabel(phoneme);
                              return (
                                <li
                                  key={phoneme.id}
                                  data-unit-level="phoneme"
                                  data-unit-index={phoneme.index}
                                >
                                  <button
                                    type="button"
                                    aria-label={`试听 Phoneme ${phoneme.index + 1} ${phone}`}
                                    onClick={() => onSeekSample?.(phoneme.refinedSample)}
                                  >
                                    <code>P{String(phoneme.index + 1).padStart(3, '0')}</code>
                                    <strong>{phone}</strong>
                                    <span>{phoneme.kind || 'phone'}</span>
                                    <small>{refinedRange(phoneme)}</small>
                                  </button>
                                </li>
                              );
                            })}
                          </ol>
                        ) : null}
                      </li>
                    );
                  })}
                </ol>
              ) : null}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
