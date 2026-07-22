import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { Brand } from '../components/Brand';
import { ChartPlayer } from '../components/ChartPlayer';
import type { ReferenceChartGroup } from '../types';
import { formatTime } from '../utils/time';

const GROUPS: Array<{ value: '' | ReferenceChartGroup; label: string }> = [
  { value: '', label: '全部' },
  { value: 'SPEED_CLUB', label: 'CLUB' },
  { value: 'SPEED_DEVIL', label: 'DEVIL' },
  { value: 'SPEED_REMIX', label: 'REMIX' },
];

export function ChartEnginePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState('');
  const [group, setGroup] = useState<'' | ReferenceChartGroup>('');
  const listQuery = useQuery({
    queryKey: ['reference-charts', 'pump-single', group, search.trim()],
    queryFn: () => api.getReferenceCharts({ mode: 'pump-single', group, search: search.trim() }),
  });
  const statisticsQuery = useQuery({
    queryKey: ['chart-corpus-statistics'],
    queryFn: api.getChartCorpusStatistics,
    staleTime: Infinity,
  });
  const selectedFromUrl = searchParams.get('chart');
  const selectedChartId = selectedFromUrl || listQuery.data?.items[0]?.id || '';
  const chartQuery = useQuery({
    queryKey: ['reference-chart', selectedChartId],
    queryFn: () => api.getReferenceChart(selectedChartId),
    enabled: Boolean(selectedChartId),
    staleTime: Infinity,
  });
  const selectedSummary = useMemo(
    () => listQuery.data?.items.find((item) => item.id === selectedChartId),
    [listQuery.data?.items, selectedChartId],
  );
  const chart = chartQuery.data;

  const chooseChart = (chartId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('chart', chartId);
    setSearchParams(next);
  };

  return (
    <main className="chart-engine-shell">
      <header className="chart-engine-header">
        <div className="chart-engine-header-left">
          <Brand />
          <div className="header-divider" />
          <div><span className="eyebrow">REAL SPEED CORPUS</span><h1>AI Chart Engine</h1></div>
        </div>
        <nav aria-label="Chart Engine 导航">
          <Link className="toolbar-button" to="/">歌曲工作区</Link>
          <span className="real-data-badge">● 本地授权语料</span>
        </nav>
      </header>

      <section className="chart-corpus-strip" aria-label="真实谱面语料统计">
        <div><small>ALL CHARTS</small><strong>{statisticsQuery.data?.chartCount ?? listQuery.data?.corpusTotal ?? '—'}</strong></div>
        <div><small>FIVE-LANE</small><strong>{statisticsQuery.data?.singleChartCount ?? listQuery.data?.total ?? '—'}</strong></div>
        <div><small>SONGS</small><strong>{statisticsQuery.data?.songCount ?? '—'}</strong></div>
        <div><small>TOTAL NOTES</small><strong>{statisticsQuery.data?.totalNotes.toLocaleString() ?? '—'}</strong></div>
        <div><small>GROUPS</small><strong>CLUB · DEVIL · REMIX</strong></div>
      </section>

      <div className="chart-library-layout">
        <aside className="reference-library-panel" aria-label="本地五轨参考谱面库">
          <div className="chart-panel-heading">
            <span className="eyebrow">REFERENCE LIBRARY</span>
            <h2>本地五轨参考谱面</h2>
            <p>直接读取本地授权的 SPEED SM 与配套 MP3，不使用演示或占位数据。</p>
          </div>
          <label className="chart-library-search">
            <span>⌕</span>
            <input aria-label="搜索真实谱面" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="歌名或文件名" />
          </label>
          <div className="chart-group-tabs" aria-label="谱面分组">
            {GROUPS.map((option) => (
              <button key={option.value || 'all'} className={group === option.value ? 'active' : ''} onClick={() => setGroup(option.value)}>{option.label}</button>
            ))}
          </div>
          <div className="reference-chart-count">
            <strong>{listQuery.data?.total ?? 0}</strong> 个 pump-single 结果
          </div>
          <div className="reference-chart-list">
            {listQuery.isLoading ? <div className="reference-list-state"><span className="spinner" /> 正在索引真实谱面…</div> : null}
            {listQuery.isError ? <div className="reference-list-state error-state"><strong>本地参考语料库不可用</strong><p>{listQuery.error.message}</p><button onClick={() => listQuery.refetch()}>重试</button></div> : null}
            {listQuery.data?.items.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`reference-chart-item${selectedChartId === item.id ? ' active' : ''}`}
                aria-pressed={selectedChartId === item.id}
                onClick={() => chooseChart(item.id)}
              >
                <span><small>{item.group.replace('SPEED_', '')}</small><b>Lv.{item.meter}</b></span>
                <strong>{item.title}</strong>
                <p>{item.bpm.toFixed(item.bpm % 1 ? 1 : 0)} BPM · {formatTime(item.durationSec).slice(0, -4)} · {item.noteCount.toLocaleString()} notes</p>
              </button>
            ))}
            {listQuery.data && !listQuery.data.items.length ? <div className="reference-list-state">没有匹配的真实谱面。</div> : null}
          </div>
        </aside>

        <section className="reference-preview-workspace">
          {chartQuery.isLoading ? <div className="chart-full-state"><span className="spinner large" /><strong>正在解析 SM 与绝对时间…</strong></div> : null}
          {chartQuery.isError ? <div className="chart-full-state error-state"><strong>无法载入谱面</strong><p>{chartQuery.error.message}</p><button onClick={() => chartQuery.refetch()}>重试</button></div> : null}
          {chart ? (
            <>
              <header className="reference-chart-header">
                <div><span className="eyebrow">{chart.sourceGroup} / {chart.mode}</span><h2>{chart.title}</h2><p>{chart.artist || '未知艺术家'} · {chart.music}</p></div>
                <div className="reference-chart-meter"><small>{chart.difficulty}</small><strong>Lv.{chart.meter}</strong></div>
              </header>
              <div className="reference-chart-facts">
                <span><small>BPM</small><strong>{chart.bpm.toFixed(chart.bpm % 1 ? 1 : 0)}</strong></span>
                <span><small>OFFSET</small><strong>{chart.offsetSec >= 0 ? '+' : ''}{chart.offsetSec.toFixed(3)} s</strong></span>
                <span><small>TEMPO MAP</small><strong>{chart.tempoMap.length} 段</strong></span>
                <span><small>DURATION</small><strong>{formatTime(chart.durationSec).slice(0, -4)}</strong></span>
                <span><small>EVENTS</small><strong>{chart.statistics?.eventCount.toLocaleString() ?? chart.events.length.toLocaleString()}</strong></span>
                <span><small>NPS PEAK</small><strong>{chart.statistics?.npsPeak.toFixed(2) ?? '—'}</strong></span>
              </div>
              <ChartPlayer
                key={chart.id}
                chart={chart}
                audioUrl={api.referenceChartAudioUrl(chart.id)}
                durationSec={selectedSummary?.durationSec ?? chart.durationSec}
              />
            </>
          ) : !chartQuery.isLoading && !chartQuery.isError ? (
            <div className="chart-full-state"><strong>选择一首真实谱面开始预览</strong><p>播放器会以解析后的绝对 timeSec 同步原始 MP3。</p></div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
