import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { EditorQuickStart } from '../src/components/EditorQuickStart';

describe('EditorQuickStart', () => {
  it('shows the four-step first-run workflow in product order and opens lyric alignment', async () => {
    const user = userEvent.setup();
    const startLyrics = vi.fn();

    render(
      <EditorQuickStart
        open
        onClose={vi.fn()}
        onStartEditing={vi.fn()}
        onStartLyrics={startLyrics}
      />,
    );

    expect(screen.getByRole('dialog', { name: '先告诉卡点工坊：你想卡什么？' })).toBeVisible();
    const workflow = screen.getByRole('list', { name: '首次使用流程' });
    expect(within(workflow).getAllByRole('listitem').map((step) => (
      within(step).getByRole('strong').textContent
    ))).toEqual([
      '导入歌曲',
      'AI 分析人声',
      '查看 Mora 时间轴',
      '导出制谱数据',
    ]);
    await user.click(screen.getByRole('button', { name: '开始人声歌词卡点' }));
    expect(startLyrics).toHaveBeenCalledOnce();
  });

  it('closes with Escape', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <EditorQuickStart
        open
        onClose={onClose}
        onStartEditing={vi.fn()}
        onStartLyrics={vi.fn()}
      />,
    );

    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('does not render while closed', () => {
    render(
      <EditorQuickStart
        open={false}
        onClose={vi.fn()}
        onStartEditing={vi.fn()}
        onStartLyrics={vi.fn()}
      />,
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});
