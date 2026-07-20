import { Link } from 'react-router-dom';
import type { ProjectSummary } from '../types';
import { formatTime } from '../utils/time';

const statusLabels = {
  unprocessed: '未分析',
  processing: '分析中',
  completed: '已完成',
  edited: '已编辑',
  failed: '失败',
} as const;

function projectMetric(project: ProjectSummary, key: 'bpm' | 'durationSec' | 'hitPointCount') {
  if (project[key] !== undefined) return project[key];
  if (key === 'bpm') return project.track?.tempoMap?.[0]?.bpm;
  if (key === 'durationSec') return project.track?.durationSec;
  return project.track?.hitPoints?.length;
}

export function ProjectCard({ project, featured = false }: { project: ProjectSummary; featured?: boolean }) {
  const bpm = projectMetric(project, 'bpm');
  const duration = projectMetric(project, 'durationSec');
  const hitCount = projectMetric(project, 'hitPointCount');
  const mode = project.analysisMode ?? project.track?.analysis?.mode;
  return (
    <article className={`project-card${featured ? ' project-card-featured' : ''}`} data-project-title={project.title}>
      <Link className="project-cover" to={`/projects/${project.id}`} aria-label={`打开 ${project.title}`}>
        {project.coverUrl ? <img src={project.coverUrl} alt="" /> : null}
        <span className="cover-fallback" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="cover-play">▶</span>
      </Link>
      <div className="project-body">
        <div className="project-heading">
          <div>
            <h3>{project.title}</h3>
            <p>{project.artist || '未知艺术家'}</p>
          </div>
          <span className={`status-pill status-${project.status}`}>{statusLabels[project.status]}</span>
        </div>
        <div className="project-tags">
          <span>{project.genre || '未分类'}</span>
          {mode ? <span>{mode}</span> : null}
        </div>
        <dl className="project-metrics">
          <div><dt>BPM</dt><dd>{typeof bpm === 'number' ? bpm.toFixed(bpm % 1 ? 1 : 0) : '—'}</dd></div>
          <div><dt>时长</dt><dd>{typeof duration === 'number' ? formatTime(duration).slice(0, -4) : '—'}</dd></div>
          <div><dt>击打点</dt><dd>{typeof hitCount === 'number' ? hitCount : '—'}</dd></div>
        </dl>
        <div className="project-footer">
          <small>更新于 {new Date(project.updatedAt).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</small>
          <Link className="text-button" to={`/projects/${project.id}`}>打开编辑器 <span>→</span></Link>
        </div>
      </div>
    </article>
  );
}
