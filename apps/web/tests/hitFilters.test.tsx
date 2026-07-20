import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { HitFilters } from '../src/components/HitFilters';
import { resetEditorStore, useEditorStore } from '../src/state/editorStore';

describe('HitFilters candidate layer', () => {
  beforeEach(() => resetEditorStore());

  it('selects a candidate lane without changing hit-point data', () => {
    render(<HitFilters />);

    fireEvent.click(screen.getByRole('button', { name: 'Melody' }));

    expect(useEditorStore.getState().filters.candidateLane).toBe('melody');
    expect(screen.getByRole('button', { name: 'Melody' })).toHaveClass('active');
    expect(screen.getByTitle('accepted')).toBeInTheDocument();
    expect(screen.getByTitle('uncertain')).toBeInTheDocument();
    expect(screen.getByTitle('alternative')).toBeInTheDocument();
  });

  it('can hide candidate events independently of final hit points', () => {
    render(<HitFilters />);
    fireEvent.click(screen.getByText('显示选项⌄'));
    fireEvent.click(screen.getByRole('checkbox', { name: '候选事件' }));

    expect(useEditorStore.getState().filters.showCandidateEvents).toBe(false);
    expect(useEditorStore.getState().filters.showHitPoints).toBe(true);
  });
});
