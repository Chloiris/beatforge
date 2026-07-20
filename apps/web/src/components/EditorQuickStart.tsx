import { useEffect } from 'react';

interface EditorQuickStartProps {
  open: boolean;
  onClose: () => void;
  onStartLyrics: () => void;
  onStartEditing: () => void;
}

export function EditorQuickStart({
  open,
  onClose,
  onStartLyrics,
  onStartEditing,
}: EditorQuickStartProps) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div className="editor-guide-backdrop">
      <section
        aria-labelledby="editor-guide-title"
        aria-modal="true"
        className="editor-guide-dialog"
        role="dialog"
      >
        <button
          aria-label="关闭使用引导"
          className="editor-guide-close"
          onClick={onClose}
          type="button"
        >
          ×
        </button>
        <header>
          <span>QUICK START</span>
          <h2 id="editor-guide-title">先告诉卡点工坊：你想卡什么？</h2>
          <p>不用先理解所有参数。选一个目标，按下面四步完成第一轮。</p>
        </header>

        <div className="editor-guide-paths">
          <article className="recommended">
            <i>推荐 · 日语歌曲</i>
            <strong>人声发音</strong>
            <p>粘贴准确歌词，让本地模型按发音对齐，再约束到 1/16 网格。</p>
            <button onClick={onStartLyrics} type="button">开始人声歌词卡点</button>
          </article>
          <article>
            <i>已有自动分析结果</i>
            <strong>鼓 / 钢琴 / 乐器</strong>
            <p>在“分轨与音源”里只显示目标声部，试听后保留或修正候选点。</p>
            <button onClick={onStartEditing} type="button">直接编辑已有点</button>
          </article>
        </div>

        <ol className="editor-guide-steps" aria-label="首次使用流程">
          <li><b>1</b><span><strong>导入歌曲</strong>添加本地音频并创建项目。</span></li>
          <li><b>2</b><span><strong>AI 分析人声</strong>使用 Japanese HuBERT CTC 生成发音事件。</span></li>
          <li><b>3</b><span><strong>查看 Mora 时间轴</strong>检查每个发音单位的采样位置与置信度。</span></li>
          <li><b>4</b><span><strong>导出制谱数据</strong>确认候选点后导出 JSON 或 CSV。</span></li>
        </ol>

        <footer>
          <span><kbd>Space</kbd> 播放/暂停</span>
          <span><kbd>双击</kbd> 添加点</span>
          <span><kbd>Delete</kbd> 删除</span>
          <span><kbd>Ctrl/⌘ Z</kbd> 撤销</span>
          <small>关闭后可随时点击顶部“使用引导”重新打开</small>
        </footer>
      </section>
    </div>
  );
}
