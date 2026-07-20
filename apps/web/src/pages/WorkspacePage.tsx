import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Brand } from '../components/Brand';
import { ProjectCard } from '../components/ProjectCard';
import { UploadDialog } from '../components/UploadDialog';
import type { ProjectStatus } from '../types';

const statusOptions: Array<{ value: '' | ProjectStatus; label: string }> = [
  { value: '', label: '全部' },
  { value: 'unprocessed', label: '未分析' },
  { value: 'processing', label: '分析中' },
  { value: 'completed', label: '已完成' },
  { value: 'edited', label: '已编辑' },
];

export function WorkspacePage() {
  const [uploadOpen, setUploadOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<'' | ProjectStatus>('');
  const projectsQuery = useQuery({ queryKey: ['projects'], queryFn: () => api.getProjects() });
  const projects = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase('zh-CN');
    return (projectsQuery.data?.items ?? []).filter((project) =>
      (!needle || `${project.title} ${project.artist} ${project.genre}`.toLocaleLowerCase('zh-CN').includes(needle)) &&
      (!status || project.status === status),
    );
  }, [projectsQuery.data, search, status]);
  const demoTitles = new Set(['霓虹脉冲', '钢铁断层', '玻璃潮汐']);
  const demos = projects.filter((project) => project.coverUrl.includes('/api/assets/covers/') || demoTitles.has(project.title));
  const demoIds = new Set(demos.map((project) => project.id));
  const recent = projects.filter((project) => !demoIds.has(project.id));

  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <Brand />
        <div className="header-divider" />
        <div className="workspace-title"><span>LIBRARY / 本地工作区</span><h1>歌曲工作区</h1></div>
        <button className="primary-button import-button" onClick={() => setUploadOpen(true)}><span>＋</span> 导入音频</button>
      </header>
      <section className="workspace-hero">
        <div>
          <div className="eyebrow">AI-ASSISTED CHART WORKSTATION</div>
          <h2>把每一次发音，<br /><em>落在正确的采样点。</em></h2>
          <p>从人声 Mora 到鼓点瞬态，在专业时间轴中检查声学证据、编辑候选事件并导出制谱数据。</p>
        </div>
        <button className="hero-upload-card" onClick={() => setUploadOpen(true)}>
          <span className="hero-upload-icon">↥</span>
          <strong>导入一首歌曲</strong>
          <small>支持 WAV、FLAC、MP3、M4A、AAC 与 OGG</small>
          <i>开始分析 <span>→</span></i>
        </button>
      </section>
      <section className="library-section">
        <div className="section-heading">
          <div><span className="eyebrow">DEMO SESSIONS</span><h2>演示项目</h2></div>
          <p>三首可复现的本地合成歌曲，用于验证不同节奏特征。</p>
        </div>
        {projectsQuery.isLoading ? (
          <div className="project-grid"><div className="card-skeleton" /><div className="card-skeleton" /><div className="card-skeleton" /></div>
        ) : projectsQuery.isError ? (
          <div className="empty-state error-state"><strong>无法连接本地分析服务</strong><p>{projectsQuery.error.message}</p><button onClick={() => projectsQuery.refetch()}>重试</button></div>
        ) : demos.length ? (
          <div className="project-grid">{demos.map((project) => <ProjectCard key={project.id} project={project} featured />)}</div>
        ) : (
          <div className="empty-state"><strong>还没有演示项目</strong><p>请先运行 <code>python scripts/beatforge.py seed</code> 生成三首本地合成音频。</p></div>
        )}
      </section>
      <section className="library-section recent-section">
        <div className="section-heading toolbar-heading">
          <div><span className="eyebrow">RECENT PROJECTS</span><h2>最近项目</h2></div>
          <div className="library-controls">
            <label className="search-box"><span>⌕</span><input aria-label="搜索歌曲" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索歌曲、艺术家或风格" /></label>
            <div className="filter-tabs" aria-label="状态筛选">{statusOptions.map((option) => <button key={option.value || 'all'} className={status === option.value ? 'active' : ''} onClick={() => setStatus(option.value)}>{option.label}</button>)}</div>
          </div>
        </div>
        {recent.length ? (
          <div className="project-grid recent-grid">{recent.map((project) => <ProjectCard key={project.id} project={project} />)}</div>
        ) : !projectsQuery.isLoading ? (
          <div className="empty-state"><strong>{search || status ? '没有匹配的最近项目' : '暂无最近导入项目'}</strong><p>{search || status ? '调整关键词或状态筛选。' : '导入一首本地歌曲后，它会出现在这里。'}</p></div>
        ) : null}
      </section>
      <footer className="workspace-footer"><span>BEATFORGE STUDIO / LOCAL-FIRST</span><span>时间基准：原始音频整数采样点</span></footer>
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </main>
  );
}
