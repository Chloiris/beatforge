import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import type {
  AlignmentHierarchyUnit,
  AlignmentLayer,
  AlignmentMethod,
  AlignmentMethodId,
  AlignmentResult,
  AlignmentResultStatus,
  StemKind,
  TrackDetail,
  WaveformPeaks,
} from '../types';
import { cssVar } from '../utils/designTokens';
import { MoraInspector } from './MoraInspector';
import {
  TimelineCanvas,
  type AlignmentTimelineLane,
  type AlignmentTimelineToken,
} from './TimelineCanvas';

const HUBERT_METHOD: AlignmentMethodId = 'ctc';
const HUBERT_COLOR = cssVar('--alignment');
const STATUS_LABELS: Record<AlignmentResultStatus, string> = {
  empty: '未运行',
  queued: '排队中',
  processing: '处理中',
  completed: '已完成',
  failed: '失败',
  unavailable: '不可用',
};
const ALIGNMENT_WAVEFORM_SOURCES: StemKind[] = ['vocals'];
const LAYER_OPTIONS: { id: AlignmentLayer; label: string; detail: string }[] = [
  { id: 'character', label: 'Character', detail: '字符' },
  { id: 'mora', label: 'Mora', detail: 'モーラ' },
  { id: 'phoneme', label: 'Phoneme', detail: '音素' },
];
const LAYER_LABELS: Record<AlignmentLayer | 'raw', string> = {
  character: 'Character',
  mora: 'Mora',
  phoneme: 'Phoneme',
  raw: 'Raw',
};

function hierarchyUnits(
  result: AlignmentResult,
  layer: AlignmentLayer,
): AlignmentHierarchyUnit[] {
  if (!result.hierarchy) return [];
  if (layer === 'character') return result.hierarchy.characters;
  if (layer === 'mora') return result.hierarchy.moras;
  return result.hierarchy.phonemes;
}

function displayedTokenCount(result: AlignmentResult | undefined, layer: AlignmentLayer): number {
  if (result?.status !== 'completed') return 0;
  return result.hierarchy ? hierarchyUnits(result, layer).length : result.tokens.length;
}

function completedRunId(result: AlignmentResult | undefined): string | null {
  return result?.status === 'completed' ? result.runId : null;
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return 'Alignment 请求失败，请检查本地运行环境。';
}

function resultErrorText(result: AlignmentResult | undefined): string {
  if (!result?.error) return '';
  return typeof result.error === 'string' ? result.error : result.error.message;
}

function statusOf(
  method: AlignmentMethod | undefined,
  result: AlignmentResult | undefined,
): AlignmentResultStatus {
  if (method && !method.available) return 'unavailable';
  return result?.status ?? 'empty';
}

async function getAlignmentResultOrEmpty(
  trackId: string,
  method: AlignmentMethodId,
): Promise<AlignmentResult | null> {
  try {
    return await api.getAlignmentResult(trackId, method);
  } catch (error) {
    if (
      error instanceof ApiError
      && error.status === 404
      && error.code === 'ALIGNMENT_RESULT_NOT_FOUND'
    ) return null;
    throw error;
  }
}

interface AlignmentLabProps {
  track: TrackDetail;
  initialWaveform?: WaveformPeaks;
  currentSample: number;
  isPlaying: boolean;
  followPlayback: boolean;
  onSeek: (sample: number) => void;
  onClose: () => void;
}

export function AlignmentLab({
  track,
  initialWaveform,
  currentSample,
  isPlaying,
  followPlayback,
  onSeek,
  onClose,
}: AlignmentLabProps) {
  const queryClient = useQueryClient();
  const [selectedLayer, setSelectedLayer] = useState<AlignmentLayer>('mora');
  const [selectedMoraId, setSelectedMoraId] = useState<string | null>(null);
  const [selectedAlignmentId, setSelectedAlignmentId] = useState<string | null>(null);

  const methodsQuery = useQuery({
    queryKey: ['alignment-methods'],
    queryFn: api.getAlignmentMethods,
    retry: false,
  });
  const selectedDescriptor = methodsQuery.data?.find((method) => method.id === HUBERT_METHOD);
  const selectedResultQuery = useQuery({
    queryKey: ['alignment-result', track.id, HUBERT_METHOD],
    queryFn: () => getAlignmentResultOrEmpty(track.id, HUBERT_METHOD),
    enabled: Boolean(track.id) && Boolean(selectedDescriptor?.available),
    retry: false,
    refetchInterval: (query: { state: { data?: AlignmentResult | null } }) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'processing' ? 700 : false;
    },
  });
  const selectedResult = selectedResultQuery.data ?? undefined;
  const completedResultRunId = completedRunId(selectedResult);
  const selectedReportQuery = useQuery({
    queryKey: ['alignment-report', track.id, HUBERT_METHOD, completedResultRunId],
    queryFn: () => api.getAlignmentReport(track.id, HUBERT_METHOD),
    enabled: Boolean(completedResultRunId),
    retry: false,
  });
  const selectedReport = selectedResult?.status === 'completed'
    && selectedReportQuery.data?.runId === selectedResult.runId
    ? selectedReportQuery.data
    : undefined;

  const runMutation = useMutation({
    mutationFn: () => api.runAlignment(track.id, HUBERT_METHOD),
    onSuccess: (result) => {
      queryClient.setQueryData(['alignment-result', track.id, result.method], result);
      void queryClient.invalidateQueries({
        queryKey: ['alignment-result', track.id, result.method],
      });
      if (result.status === 'completed') {
        void queryClient.invalidateQueries({
          queryKey: ['alignment-report', track.id, result.method, result.runId],
          exact: true,
        });
      }
    },
  });

  const selectedStatus = statusOf(selectedDescriptor, selectedResult);
  const selectedRunning = selectedStatus === 'queued' || selectedStatus === 'processing';
  const realMoras = useMemo(() => {
    if (selectedResult?.status !== 'completed' || !selectedResult.hierarchy) return [];
    return selectedResult.hierarchy.moras.filter((unit) => (
      unit.level === 'mora'
      && Number.isFinite(unit.refinedStartSample)
      && Number.isFinite(unit.refinedEndSample)
      && Number.isFinite(unit.refinedSample)
      && unit.refinedEndSample > unit.refinedStartSample
    ));
  }, [selectedResult]);
  const firstRealMoraId = realMoras[0]?.id ?? null;

  useEffect(() => {
    setSelectedMoraId(firstRealMoraId);
    setSelectedAlignmentId(firstRealMoraId);
  }, [completedResultRunId, firstRealMoraId]);

  const selectedMora = realMoras.find((unit) => unit.id === selectedMoraId) ?? null;

  const selectAlignmentToken = (token: AlignmentTimelineToken) => {
    setSelectedAlignmentId(token.id);
    if (!('level' in token) || !selectedResult?.hierarchy) return;
    const mora = token.level === 'mora'
      ? realMoras.find((unit) => unit.id === token.id)
      : token.level === 'character'
        ? realMoras.find((unit) => unit.characterIndices.includes(token.index))
        : realMoras.find((unit) => unit.phonemeIndices.includes(token.index));
    if (mora) setSelectedMoraId(mora.id);
  };

  const alignmentLanes = useMemo<AlignmentTimelineLane[]>(() => {
    const completed = selectedResult?.status === 'completed' ? selectedResult : undefined;
    const level = completed?.hierarchy ? selectedLayer : 'raw';
    return [{
      method: HUBERT_METHOD,
      label: `${selectedDescriptor?.name ?? 'Japanese HuBERT CTC'} · ${LAYER_LABELS[level]}`,
      color: HUBERT_COLOR,
      level,
      // Partial, failed, grid and hit-point data never become display tokens.
      // Character/mora spans are never inferred from flat phoneme strings.
      tokens: completed
        ? completed.hierarchy ? hierarchyUnits(completed, selectedLayer) : completed.tokens
        : [],
    }];
  }, [selectedDescriptor?.name, selectedLayer, selectedResult]);

  const runError = runMutation.isError ? errorText(runMutation.error) : '';
  const selectedReadError = selectedResultQuery.isError
    ? errorText(selectedResultQuery.error)
    : '';
  const selectedOperationError = runError || selectedReadError || resultErrorText(selectedResult);

  return (
    <section
      id="vocal-alignment-panel"
      className="alignment-lab"
      data-testid="alignment-lab"
      aria-labelledby="vocal-alignment-title"
    >
      <header className="alignment-controls">
        <div className="alignment-fixed-method">
          <i />
          <span>
            <strong id="vocal-alignment-title">Vocal Alignment</strong>
            <small>Japanese HuBERT CTC · 人声日语音素对齐</small>
          </span>
        </div>
        <button
          className="primary-button alignment-run-button"
          type="button"
          onClick={() => runMutation.mutate()}
          disabled={!selectedDescriptor?.available || selectedRunning || runMutation.isPending}
        >
          {runMutation.isPending
            ? <><span className="spinner" /> 正在启动…</>
            : selectedRunning ? '本地任务运行中' : '运行 Japanese HuBERT CTC'}
        </button>
        <span className={`alignment-current-status status-${selectedStatus}`}>
          {STATUS_LABELS[selectedStatus]}
        </span>
        <span className="alignment-controls-spacer" />
        <code>{track.originalSampleRate.toLocaleString()} Hz · {track.sampleCount.toLocaleString()} samples</code>
        <button
          className="icon-button"
          type="button"
          aria-label="关闭 Vocal Alignment"
          title="关闭 Vocal Alignment"
          onClick={onClose}
        >×</button>
      </header>

      <div className="alignment-layer-bar">
        <div>
          <span className="eyebrow">VOCAL ALIGNMENT LAYER</span>
          <strong>{LAYER_LABELS[selectedLayer]}</strong>
        </div>
        <div className="alignment-layer-switch" role="group" aria-label="Vocal Alignment Layer">
          {LAYER_OPTIONS.map((layer) => (
            <button
              key={layer.id}
              type="button"
              aria-pressed={selectedLayer === layer.id}
              className={selectedLayer === layer.id ? 'active' : ''}
              onClick={() => setSelectedLayer(layer.id)}
            >
              <span>{layer.label}</span>
              <small>{layer.detail}</small>
            </button>
          ))}
        </div>
        {selectedResult?.status === 'completed' && selectedResult.hierarchy ? (
          <code>
            {selectedResult.hierarchy.characters.length.toLocaleString()} characters · {' '}
            {selectedResult.hierarchy.moras.length.toLocaleString()} moras · {' '}
            {selectedResult.hierarchy.phonemes.length.toLocaleString()} phonemes
          </code>
        ) : (
          <span className="alignment-layer-legend">
            □ aligned span / ○ alignedSample → ■ refined span / ● refinedSample
          </span>
        )}
      </div>

      {methodsQuery.isLoading ? (
        <div className="alignment-notice"><span className="spinner" /> 正在检查本地 alignment 方法…</div>
      ) : methodsQuery.isError ? (
        <div className="error-banner alignment-banner" role="alert">
          {errorText(methodsQuery.error)}
          <button type="button" onClick={() => void methodsQuery.refetch()}>重试</button>
        </div>
      ) : !selectedDescriptor ? (
        <div className="alignment-notice">后端没有提供 Japanese HuBERT CTC。</div>
      ) : null}

      {selectedDescriptor && !selectedDescriptor.available ? (
        <div className="alignment-unavailable-note">
          <strong>Japanese HuBERT CTC 当前不可用</strong>
          <span>{selectedDescriptor.reason || '本地依赖或模型尚未准备完成。'}</span>
        </div>
      ) : null}
      {selectedOperationError ? (
        <div className="error-banner alignment-banner" role="alert">{selectedOperationError}</div>
      ) : null}
      {selectedResult?.warnings.length ? (
        <div className="alignment-warning">{selectedResult.warnings.join('；')}</div>
      ) : null}
      {selectedResult?.status === 'completed' && !selectedResult.hierarchy ? (
        <div className="alignment-layer-note">
          Japanese HuBERT CTC 未提供 Character/Mora 层级；当前保留并显示原始 token，不会推测字符时间。
        </div>
      ) : selectedResult?.status === 'completed'
        && selectedResult.hierarchy
        && hierarchyUnits(selectedResult, selectedLayer).length === 0 ? (
          <div className="alignment-layer-note">本次真实对齐结果没有 {LAYER_LABELS[selectedLayer]} 区间。</div>
        ) : null}

      <div className="alignment-workspace">
        <div className="alignment-timeline-column">
          <div className="alignment-hubert-status" aria-label="Japanese HuBERT CTC 状态">
            <i />
            <span><strong>Japanese HuBERT CTC</strong><small>{STATUS_LABELS[selectedStatus]}</small></span>
            <code>{selectedResult?.status === 'completed'
              ? `${displayedTokenCount(selectedResult, selectedLayer).toLocaleString()} ${LAYER_LABELS[selectedLayer]}`
              : '—'}</code>
          </div>
          {selectedRunning ? (
            <div className="alignment-progress" aria-live="polite">
              <span className="spinner" />
              <div><strong>Japanese HuBERT CTC 正在本地运行</strong><small>结果接口每 700 ms 轮询。</small></div>
            </div>
          ) : selectedStatus === 'empty' ? (
            <div className="alignment-empty-note">运行后，真实 HuBERT Mora 会显示在下方 lane。这里不会生成占位 timestamp。</div>
          ) : null}
          <TimelineCanvas
            track={track}
            initialWaveform={initialWaveform}
            currentSample={currentSample}
            isPlaying={isPlaying}
            followPlayback={followPlayback}
            onSeek={onSeek}
            mode="alignment"
            alignmentLanes={alignmentLanes}
            waveformSources={ALIGNMENT_WAVEFORM_SOURCES}
            selectedAlignmentId={selectedAlignmentId}
            onAlignmentSelect={selectAlignmentToken}
          />
        </div>

        <MoraInspector
          mora={selectedMora}
          hierarchy={selectedResult?.hierarchy}
          candidates={track.candidateEvents}
          alignmentRunId={completedResultRunId}
          sampleRate={track.originalSampleRate}
          exportUrl={api.exportUrl(track.id, 'json')}
          onAlign={onSeek}
          report={selectedReport}
          reportLoading={selectedReportQuery.isLoading}
          reportError={selectedResult?.status === 'completed' && selectedReportQuery.isError
            ? errorText(selectedReportQuery.error)
            : ''}
        />
      </div>
    </section>
  );
}
