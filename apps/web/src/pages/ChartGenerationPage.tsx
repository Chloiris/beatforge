import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import { Brand } from '../components/Brand';
import { ChartPlayer } from '../components/ChartPlayer';
import { ChartValidationPanel } from '../components/ChartValidationPanel';
import type { ChartDocument } from '../types';
import { createGenerationSeedSequence } from '../utils/generationSeed';
import { formatTime } from '../utils/time';

const DIFFICULTY_LEVELS = Array.from({ length: 15 }, (_, index) => index + 1);

export function ChartGenerationPage() {
  const { projectId = '' } = useParams();
  const queryClient = useQueryClient();
  const [difficulty, setDifficulty] = useState(8);
  const difficultyInitialized = useRef(false);
  const [enableSpin, setEnableSpin] = useState(false);
  const [generatedChart, setGeneratedChart] = useState<ChartDocument | null>(null);
  const [takeGenerationSeed] = useState<() => number>(() => createGenerationSeedSequence());
  const projectQuery = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
    enabled: Boolean(projectId),
  });
  const track = projectQuery.data?.track ?? null;
  const latestChartQuery = useQuery({
    queryKey: ['latest-chart', track?.id],
    queryFn: () => api.getLatestChart(track!.id),
    enabled: Boolean(track?.id),
    retry: false,
  });
  useEffect(() => {
    if (!difficultyInitialized.current && latestChartQuery.data) {
      setDifficulty(latestChartQuery.data.meter);
      difficultyInitialized.current = true;
    }
  }, [latestChartQuery.data]);
  const generateMutation = useMutation({
    mutationFn: () => api.generateChart(track!.id, {
      difficulty,
      enableSpin,
      useLocalModel: true,
      seed: takeGenerationSeed(),
    }),
    onSuccess: (response) => {
      setGeneratedChart(response.chart);
      queryClient.setQueryData(['latest-chart', track?.id], response.chart);
    },
  });
  const chart = generatedChart ?? latestChartQuery.data ?? null;
  const project = projectQuery.data;
  const analysisReady = Boolean(track?.tempoMap.length && (track.hitPoints.length || track.candidateEvents.length));
  const latestMissing = latestChartQuery.error instanceof ApiError
    && latestChartQuery.error.code === 'CHART_NOT_GENERATED';

  if (projectQuery.isLoading) return <div className="full-page-state"><span className="spinner large" /><strong>正在加载工程…</strong></div>;
  if (projectQuery.isError || !project) return <div className="full-page-state error-state"><strong>无法打开工程</strong><p>{projectQuery.error?.message ?? '工程不存在'}</p><Link to="/">返回歌曲工作区</Link></div>;
  if (!track) return <div className="full-page-state"><strong>工程还没有音轨</strong><Link to="/">返回歌曲工作区</Link></div>;

  return (
    <main className="chart-generation-shell">
      <header className="chart-generation-header">
        <div className="chart-engine-header-left">
          <Brand compact />
          <Link className="back-link" to={`/projects/${project.id}`}>← 声学编辑器</Link>
          <div><span className="eyebrow">AI CHART WORKSPACE</span><h1>{project.title}</h1><p>{project.artist || '未知艺术家'} · {track.originalFileName}</p></div>
        </div>
        <nav aria-label="AI Chart Workspace 导航">
          <Link className="toolbar-button" to="/chart-engine">真实谱面库</Link>
          {chart ? <a className="primary-button chart-sm-export" href={api.chartExportUrl(track.id, chart.id)} download>导出 SM</a> : null}
        </nav>
      </header>

      <section className="chart-song-strip" aria-label="歌曲信息">
        <div><small>SONG</small><strong>{project.title}</strong></div>
        <div><small>BPM</small><strong>{track.tempoMap[0]?.bpm.toFixed(2).replace(/\.00$/, '') ?? '—'}</strong></div>
        <div><small>DURATION</small><strong>{formatTime(track.durationSec).slice(0, -4)}</strong></div>
        <div><small>CANDIDATES</small><strong>{track.candidateEvents.length.toLocaleString()}</strong></div>
        <div><small>HIT POINTS</small><strong>{track.hitPoints.length.toLocaleString()}</strong></div>
      </section>

      <div className="chart-generation-layout">
        <aside className="chart-generation-controls" aria-label="AI 制谱参数">
          <div className="chart-panel-heading"><span className="eyebrow">GENERATION PARAMETERS</span><h2>制谱参数</h2><p>统一难度控制密度与技巧；转圈作为独立身体方向开关。</p></div>
          <label className="difficulty-control">
            <span><strong>难度</strong><small>LEVEL 1–15</small></span>
            <select
              aria-label="谱面难度"
              value={difficulty}
              onChange={(event) => {
                difficultyInitialized.current = true;
                setDifficulty(Number(event.target.value));
              }}
            >
              {DIFFICULTY_LEVELS.map((level) => <option key={level} value={level}>Lv.{level}</option>)}
            </select>
          </label>
          <div className="spin-control">
            <span><strong>转圈</strong><small>五键大圈与三键小圈</small></span>
            <button
              type="button"
              role="switch"
              aria-label="启用转圈"
              aria-checked={enableSpin}
              className={`toggle-switch${enableSpin ? ' on' : ''}`}
              onClick={() => setEnableSpin((enabled) => !enabled)}
            ><i /></button>
            <b>{enableSpin ? 'ON' : 'OFF'}</b>
          </div>
          <div className="generation-source-note"><i />基于本地授权的五轨参考语料统计与当前 BeatForge 声学候选生成。</div>
          {!analysisReady ? <div className="generation-requirement">请先在声学编辑器完成 BeatForge 分析，生成候选事件后再制谱。</div> : null}
          {generateMutation.isError ? <div className="error-banner" role="alert">{generateMutation.error.message}</div> : null}
          <button
            className="primary-button generate-chart-button"
            type="button"
            disabled={!analysisReady || generateMutation.isPending}
            onClick={() => generateMutation.mutate()}
          >{generateMutation.isPending ? <><span className="spinner" /> 正在生成与验证…</> : chart ? '重新生成谱面' : '生成 AI 谱面'}</button>
          {generateMutation.data ? (
            <div className="generation-corpus-result">
              <small>REFERENCE CORPUS</small>
              <strong>{generateMutation.data.referenceCorpus.chartCount} charts / {generateMutation.data.referenceCorpus.songCount} songs</strong>
              <span>{generateMutation.data.referenceCorpus.source}</span>
              <span className="generation-model-result">
                Local Transformer · {generateMutation.data.referenceCorpus.model.used
                  ? '已参与生成'
                  : generateMutation.data.referenceCorpus.model.available
                    ? '未产生可用预测，已使用统计后备'
                    : '未安装，已使用统计后备'}
              </span>
            </div>
          ) : null}
        </aside>

        <section className="generated-chart-preview" aria-label="生成谱面预览">
          {latestChartQuery.isLoading && !chart ? <div className="chart-full-state"><span className="spinner" /><strong>正在读取最近生成谱面…</strong></div> : null}
          {latestChartQuery.isError && !latestMissing && !chart ? <div className="chart-full-state error-state"><strong>无法读取谱面</strong><p>{latestChartQuery.error.message}</p></div> : null}
          {chart ? (
            <>
              <header className="generated-chart-heading"><div><span className="eyebrow">{chart.generator.replaceAll('_', ' ')}</span><h2>{chart.title}</h2>{chart.modelProvenance ? <p>Local Transformer · {chart.modelProvenance.architecture ?? 'checkpoint'} · 本地授权数据</p> : <p>Reference corpus statistical generator</p>}</div><strong>Lv.{chart.meter}</strong></header>
              <ChartPlayer key={chart.id} chart={chart} audioUrl={api.audioUrl(track.id)} durationSec={track.durationSec} />
            </>
          ) : !latestChartQuery.isLoading || latestMissing ? (
            <div className="chart-full-state chart-empty-preview"><span className="empty-preview-lanes" aria-hidden="true"><i /><i /><i /><i /><i /></span><strong>准备生成第一张谱面</strong><p>真实 BeatForge 候选会被分配到五个身体面板，并经过密度与可玩性验证。</p></div>
          ) : null}
        </section>

        <ChartValidationPanel chart={chart} />
      </div>
    </main>
  );
}
