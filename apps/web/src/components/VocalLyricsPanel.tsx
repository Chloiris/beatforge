import { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import type {
  AlignmentResult,
  LyricsInputFormat,
  VocalLyrics,
  VocalLyricsJob,
  VocalLyricsStage,
  VocalLyricsStatus,
} from '../types';
import { VocalLyricsHierarchy } from './VocalLyricsHierarchy';

const HUBERT_METHOD = 'ctc' as const;

const formatLabels: Record<LyricsInputFormat, string> = {
  japanese: '日文原文',
  kana: '假名',
  romaji: '罗马音（需先转假名）',
  lrc: 'LRC（含时间标签）',
};

const stageLabels: Record<VocalLyricsStage, string> = {
  idle: '等待操作',
  queued: '等待本地任务',
  separating_vocals: '分离人声与伴奏',
  detecting_vocal_activity: '检测人声区段',
  transcribing: '生成本地听写草稿',
  normalizing_pronunciation: '生成假名与罗马音',
  aligning_lyrics: '对齐歌词与发音',
  refining_samples: '在原始采样点上精修',
  saving_results: '保存对齐结果',
  completed: '对齐完成',
};

const statusLabels: Record<VocalLyricsStatus, string> = {
  empty: '未录入',
  draft: '草稿',
  saved: '已保存',
  queued: '排队中',
  processing: '处理中',
  completed: '已对齐',
  failed: '失败',
};

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return '歌词操作失败，请稍后重试。';
}

function statusFromJob(job: VocalLyricsJob | undefined, lyrics: VocalLyrics | undefined): VocalLyricsStatus {
  if (!job) {
    if (!lyrics?.text.trim()) return 'empty';
    return lyrics.status === 'draft' ? 'draft' : 'saved';
  }
  if (job.status === 'queued') return 'queued';
  if (job.status === 'processing') return 'processing';
  if (job.status === 'failed') return 'failed';
  return job.result?.status ?? 'completed';
}

function statusFromAlignment(
  result: AlignmentResult | null | undefined,
  draftJob: VocalLyricsJob | undefined,
  lyrics: VocalLyrics | undefined,
): VocalLyricsStatus {
  if (result?.status === 'queued') return 'queued';
  if (result?.status === 'processing') return 'processing';
  if (result?.status === 'completed') return 'completed';
  if (result?.status === 'failed' || result?.status === 'unavailable') return 'failed';
  return statusFromJob(draftJob, lyrics);
}

async function getHubertResultOrEmpty(trackId: string): Promise<AlignmentResult | null> {
  try {
    return await api.getAlignmentResult(trackId, HUBERT_METHOD);
  } catch (error) {
    if (
      error instanceof ApiError
      && error.status === 404
      && error.code === 'ALIGNMENT_RESULT_NOT_FOUND'
    ) return null;
    throw error;
  }
}

function alignmentResultError(result: AlignmentResult | null | undefined): string {
  if (!result?.error) return '';
  return typeof result.error === 'string' ? result.error : result.error.message;
}

interface VocalLyricsPanelProps {
  trackId: string;
  defaultExpanded?: boolean;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onSeekSample?: (sample: number) => void;
}

export function VocalLyricsPanel({
  trackId,
  defaultExpanded = false,
  expanded: controlledExpanded,
  onExpandedChange,
  onSeekSample,
}: VocalLyricsPanelProps) {
  const queryClient = useQueryClient();
  const [internalExpanded, setInternalExpanded] = useState(defaultExpanded);
  const expanded = controlledExpanded ?? internalExpanded;
  const setExpanded = (value: boolean) => {
    setInternalExpanded(value);
    onExpandedChange?.(value);
  };
  const [text, setText] = useState('');
  const [inputFormat, setInputFormat] = useState<LyricsInputFormat>('japanese');
  const [draftJobId, setDraftJobId] = useState<string | null>(null);
  const [localError, setLocalError] = useState('');
  const hydratedRef = useRef<{ key: string; text: string; inputFormat: LyricsInputFormat } | null>(null);
  const refreshedAlignmentRunRef = useRef<string | null>(null);

  const lyricsQuery = useQuery({
    queryKey: ['vocal-lyrics', trackId],
    queryFn: () => api.getVocalLyrics(trackId),
    enabled: expanded && Boolean(trackId),
    retry: false,
  });
  const jobQuery = useQuery({
    queryKey: ['vocal-lyrics-job', draftJobId],
    queryFn: () => api.getVocalLyricsJob(draftJobId!),
    enabled: Boolean(draftJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'processing' || !status ? 700 : false;
    },
  });
  const alignmentQuery = useQuery({
    queryKey: ['alignment-result', trackId, HUBERT_METHOD],
    queryFn: () => getHubertResultOrEmpty(trackId),
    enabled: Boolean(trackId),
    retry: false,
    refetchInterval: (query: { state: { data?: AlignmentResult | null } }) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'processing' ? 700 : false;
    },
  });

  useEffect(() => {
    const lyrics = lyricsQuery.data;
    if (!lyrics) return;
    const hydrationKey = `${lyrics.updatedAt ?? 'empty'}:${lyrics.status}:${lyrics.anchors.length}`;
    const current = hydratedRef.current;
    const hasLocalEdits = current
      ? text !== current.text || inputFormat !== current.inputFormat
      : text.trim().length > 0;
    if (current?.key === hydrationKey || hasLocalEdits) return;
    setText(lyrics.text);
    setInputFormat(lyrics.inputFormat);
    hydratedRef.current = { key: hydrationKey, text: lyrics.text, inputFormat: lyrics.inputFormat };
  }, [inputFormat, lyricsQuery.data, text]);

  useEffect(() => {
    const job = jobQuery.data;
    if (job?.status !== 'completed') return;
    if (job.result) {
      queryClient.setQueryData(['vocal-lyrics', trackId], job.result);
    } else {
      void queryClient.invalidateQueries({ queryKey: ['vocal-lyrics', trackId] });
    }
  }, [jobQuery.data, queryClient, trackId]);

  useEffect(() => {
    const result = alignmentQuery.data;
    if (result?.status !== 'completed' || refreshedAlignmentRunRef.current === result.runId) return;
    refreshedAlignmentRunRef.current = result.runId;
    // HuBERT publication can replace vocal CandidateEvents. Refresh the project
    // only for the fixed CTC run; ASR draft completion remains separate.
    void queryClient.invalidateQueries({ queryKey: ['project'] });
  }, [alignmentQuery.data, queryClient]);

  const saveMutation = useMutation({
    mutationFn: () => api.saveVocalLyrics(trackId, text, inputFormat),
    onSuccess: (lyrics) => {
      queryClient.setQueryData(['vocal-lyrics', trackId], lyrics);
      hydratedRef.current = {
        key: `${lyrics.updatedAt ?? 'empty'}:${lyrics.status}:${lyrics.anchors.length}`,
        text: lyrics.text,
        inputFormat: lyrics.inputFormat,
      };
      setLocalError('');
    },
  });
  const alignMutation = useMutation({
    mutationFn: () => api.runAlignment(trackId, HUBERT_METHOD),
    onSuccess: (result) => {
      queryClient.setQueryData(['alignment-result', trackId, HUBERT_METHOD], result);
      void queryClient.invalidateQueries({
        queryKey: ['alignment-result', trackId, HUBERT_METHOD],
      });
    },
  });
  const draftMutation = useMutation({
    mutationFn: () => api.generateVocalLyricsDraft(trackId, inputFormat),
  });

  const storedLyrics = lyricsQuery.data;
  const hasLocalEdits = storedLyrics
    ? text !== storedLyrics.text || inputFormat !== storedLyrics.inputFormat
    : text.trim().length > 0;
  const displayLyrics = jobQuery.data?.result ?? storedLyrics;
  const alignmentResult = alignmentQuery.data;
  const alignmentHierarchy = alignmentResult?.status === 'completed'
    ? alignmentResult.hierarchy ?? null
    : null;
  const status = statusFromAlignment(alignmentResult, jobQuery.data, displayLyrics);
  const isDraftRunning = jobQuery.data?.status === 'queued' || jobQuery.data?.status === 'processing';
  const isAlignmentRunning = alignmentResult?.status === 'queued'
    || alignmentResult?.status === 'processing';
  const isOperationRunning = isDraftRunning || isAlignmentRunning;
  const stage: VocalLyricsStage = isAlignmentRunning
    ? alignmentResult?.status === 'queued' ? 'queued' : 'aligning_lyrics'
    : alignmentResult?.status === 'completed'
      ? 'completed'
      : jobQuery.data?.stage ?? 'idle';
  const metadataProgress = typeof alignmentResult?.metadata.progress === 'number'
    ? alignmentResult.metadata.progress
    : null;
  const progress = alignmentResult?.status === 'completed'
    ? 1
    : isAlignmentRunning
      ? metadataProgress ?? 0
      : jobQuery.data?.progress ?? displayLyrics?.progress ?? 0;
  const progressLabel = isAlignmentRunning && metadataProgress === null
    ? '运行中'
    : `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
  const stageLabel = alignmentResult?.status === 'completed'
    ? 'HuBERT 对齐完成'
    : jobQuery.data?.kind === 'asr_draft' && stage === 'completed'
      ? 'ASR 草稿完成'
      : stageLabels[stage];
  const characterCount = useMemo(
    () => Array.from(text).filter((character) => !/\s/u.test(character)).length,
    [text],
  );
  const operationError = localError
    || jobQuery.data?.error?.message
    || displayLyrics?.error?.message
    || (lyricsQuery.isError ? errorMessage(lyricsQuery.error) : '')
    || (saveMutation.isError ? errorMessage(saveMutation.error) : '')
    || (alignMutation.isError ? errorMessage(alignMutation.error) : '')
    || alignmentResultError(alignmentResult)
    || (alignmentQuery.isError ? errorMessage(alignmentQuery.error) : '')
    || (draftMutation.isError ? errorMessage(draftMutation.error) : '')
    || (jobQuery.isError ? errorMessage(jobQuery.error) : '');

  const save = async (): Promise<VocalLyrics | null> => {
    try {
      return await saveMutation.mutateAsync();
    } catch {
      return null;
    }
  };

  const startAlignment = async () => {
    setLocalError('');
    if (!text.trim()) {
      setLocalError('请先粘贴歌词，或使用本地 ASR 生成草稿。');
      return;
    }
    if (inputFormat === 'romaji') {
      setLocalError('罗马音存在分词与长音歧义，请先转换为日文原文或假名后对齐。');
      return;
    }
    if (hasLocalEdits && !(await save())) return;
    try {
      await alignMutation.mutateAsync();
    } catch {
      // The mutation error is rendered with the rest of the operation state.
    }
  };

  const startDraft = async () => {
    setLocalError('');
    try {
      const response = await draftMutation.mutateAsync();
      setDraftJobId(response.jobId);
    } catch {
      // The mutation error is rendered with the rest of the operation state.
    }
  };

  return (
    <section className={`vocal-lyrics-panel${expanded ? ' expanded' : ''}`}>
      <button
        className="vocal-lyrics-toggle"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
      >
        <span><i>歌</i><strong>HuBERT 歌词对齐（可选）</strong><small>Character → Mora → Phoneme</small></span>
        <span className={`lyrics-status status-${status}`}>{statusLabels[status]}</span>
        <b aria-hidden="true">{expanded ? '−' : '+'}</b>
      </button>
      {expanded ? (
        <div className="vocal-lyrics-content">
          <ol className="lyrics-workflow-steps" aria-label="人声歌词卡点步骤">
            <li className={text.trim() ? 'done' : 'active'}><b>1</b>提供准确歌词</li>
            <li className={isAlignmentRunning ? 'active' : alignmentHierarchy ? 'done' : ''}><b>2</b>HuBERT 发音对齐</li>
            <li className={alignmentHierarchy ? 'active' : ''}><b>3</b>展开检查层级</li>
          </ol>
          <div className="vocal-lyrics-body">
          <div className="lyrics-entry-column">
            <div className="lyrics-entry-head">
              <label htmlFor={`lyrics-format-${trackId}`}>输入格式</label>
              <select
                id={`lyrics-format-${trackId}`}
                aria-label="歌词输入格式"
                value={inputFormat}
                onChange={(event) => setInputFormat(event.target.value as LyricsInputFormat)}
              >
                {Object.entries(formatLabels).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
              <span>{characterCount.toLocaleString()} 字符</span>
            </div>
            <textarea
              aria-label="歌词文本"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder={'粘贴准确的日文歌词，每行可对应一个乐句…\n也支持假名或带时间标签的 LRC。'}
              spellCheck={false}
            />
            {inputFormat === 'romaji' ? (
              <p className="lyrics-format-warning">罗马音仅保存。请先转换为日文或假名，再进行发音对齐。</p>
            ) : null}
            <div className="lyrics-actions">
              <button
                className="lyrics-align-button"
                type="button"
                onClick={() => void startAlignment()}
                disabled={!text.trim() || inputFormat === 'romaji' || alignMutation.isPending || isOperationRunning}
              >
                {alignMutation.isPending ? '正在启动…' : hasLocalEdits ? '保存并运行 HuBERT' : '运行 Japanese HuBERT CTC'}
              </button>
              <button
                type="button"
                onClick={() => void save()}
                disabled={!text.trim() || !hasLocalEdits || saveMutation.isPending || isOperationRunning}
              >
                {saveMutation.isPending ? '正在保存…' : hasLocalEdits ? '仅保存歌词' : '歌词已保存'}
              </button>
              {!text.trim() ? (
                <button
                  type="button"
                  onClick={() => void startDraft()}
                  disabled={draftMutation.isPending || isOperationRunning}
                >
                  {draftMutation.isPending ? '正在启动…' : '没有歌词文本？从人声生成草稿'}
                </button>
              ) : null}
            </div>
            <p className="lyrics-privacy-note">全部在本地完成。ASR 草稿与分轨结果不会上传到云端。</p>
          </div>

          <div className="lyrics-alignment-column">
            {isOperationRunning || stage !== 'idle' ? (
              <div className="lyrics-stage-card" aria-live="polite">
                <span className={isOperationRunning ? 'spinner' : 'lyrics-stage-icon'}>{isOperationRunning ? null : '◎'}</span>
                <div><small>当前阶段</small><strong>{stageLabel}</strong></div>
                <output>{progressLabel}</output>
                <i><b style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></i>
              </div>
            ) : (
              <div className="lyrics-get-started">
                <span>1/16</span>
                <div><strong>这里不会自动猜歌词</strong><p>左侧粘贴准确歌词后，系统会在本地定位发音 sample。纯器乐段落可以跳过。</p></div>
              </div>
            )}
            {operationError ? <div className="lyrics-error" role="alert">{operationError}</div> : null}
            {alignmentHierarchy ? (
              <VocalLyricsHierarchy
                key={alignmentResult?.runId}
                hierarchy={alignmentHierarchy}
                onSeekSample={onSeekSample}
              />
            ) : alignmentResult?.status === 'completed' ? (
              <div className="lyrics-error">本次 HuBERT 结果没有 typed hierarchy；不会推测 Mora 或音素时间。</div>
            ) : null}
          </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
