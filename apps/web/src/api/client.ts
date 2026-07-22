import type {
  AlignmentMethod,
  AlignmentMethodId,
  AlignmentReport,
  AlignmentResult,
  AnalysisJob,
  AnalysisMode,
  AnalyzeResponse,
  ApiErrorBody,
  CandidateEvent,
  ChartCorpusStatistics,
  ChartDocument,
  ChartGenerationResponse,
  ChartMode,
  GenerateChartRequest,
  HitPoint,
  Project,
  ProjectDetail,
  ProjectListResponse,
  ReferenceChartGroup,
  ReferenceChartListResponse,
  LyricsInputFormat,
  StemKind,
  TempoSegment,
  TrackDetail,
  UploadResponse,
  WaveformPeaks,
  VocalLyrics,
  VocalLyricsJob,
  VocalLyricsJobResponse,
} from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api';

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: unknown;

  constructor(message: string, code = 'REQUEST_FAILED', status = 0, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(apiUrl(path), { ...init, headers });
  if (!response.ok) {
    let body: ApiErrorBody | undefined;
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      body = undefined;
    }
    throw new ApiError(
      body?.error?.message ?? `请求失败（HTTP ${response.status}）`,
      body?.error?.code,
      response.status,
      body?.error?.details,
    );
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  getReferenceCharts: (filters: {
    mode?: ChartMode;
    group?: ReferenceChartGroup | '';
    search?: string;
  } = {}) => {
    const params = new URLSearchParams();
    if (filters.mode) params.set('mode', filters.mode);
    if (filters.group) params.set('group', filters.group);
    if (filters.search) params.set('search', filters.search);
    const query = params.size ? `?${params}` : '';
    return request<ReferenceChartListResponse>(`/chart-engine/reference-charts${query}`);
  },

  getReferenceChart: (chartId: string) =>
    request<ChartDocument>(`/chart-engine/reference-charts/${encodeURIComponent(chartId)}`),

  getChartCorpusStatistics: () =>
    request<ChartCorpusStatistics>('/chart-engine/statistics'),

  generateChart: (trackId: string, input: GenerateChartRequest) =>
    request<ChartGenerationResponse>(`/tracks/${encodeURIComponent(trackId)}/chart/generate`, {
      method: 'POST',
      body: JSON.stringify(input),
    }),

  getLatestChart: (trackId: string) =>
    request<ChartDocument>(`/tracks/${encodeURIComponent(trackId)}/chart/latest`),

  getAlignmentMethods: () => request<AlignmentMethod[]>('/alignment/methods'),

  runAlignment: (trackId: string, method: AlignmentMethodId) =>
    request<AlignmentResult>(`/tracks/${encodeURIComponent(trackId)}/alignment/run`, {
      method: 'POST',
      body: JSON.stringify({ method }),
    }),

  getAlignmentResult: (trackId: string, method: AlignmentMethodId) =>
    request<AlignmentResult>(
      `/tracks/${encodeURIComponent(trackId)}/alignment/${encodeURIComponent(method)}`,
    ),

  getAlignmentReport: (trackId: string, method: AlignmentMethodId) =>
    request<AlignmentReport>(
      `/tracks/${encodeURIComponent(trackId)}/alignment/${encodeURIComponent(method)}/report`,
    ),

  getProjects: (search = '', status = '') => {
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (status) params.set('status', status);
    const query = params.size ? `?${params}` : '';
    return request<ProjectListResponse>(`/projects${query}`);
  },

  getProject: (projectId: string) => request<ProjectDetail>(`/projects/${encodeURIComponent(projectId)}`),

  createProject: (project: Partial<Project>) =>
    request<Project>('/projects', { method: 'POST', body: JSON.stringify(project) }),

  updateProject: (projectId: string, changes: Partial<Project>) =>
    request<Project>(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'PATCH',
      body: JSON.stringify(changes),
    }),

  deleteProject: (projectId: string) =>
    request<void>(`/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' }),

  uploadTrack: (file: File, metadata: { title?: string; artist?: string; genre?: string }) => {
    const body = new FormData();
    body.set('file', file);
    if (metadata.title) body.set('title', metadata.title);
    if (metadata.artist) body.set('artist', metadata.artist);
    if (metadata.genre) body.set('genre', metadata.genre);
    return request<UploadResponse>('/tracks/upload', { method: 'POST', body });
  },

  analyzeTrack: (trackId: string, mode: AnalysisMode, sensitivity: number) =>
    request<AnalyzeResponse>(`/tracks/${encodeURIComponent(trackId)}/analyze`, {
      method: 'POST',
      body: JSON.stringify({ mode, sensitivity }),
    }),

  getJob: (jobId: string) => request<AnalysisJob>(`/analysis-jobs/${encodeURIComponent(jobId)}`),

  getTrack: (trackId: string) => request<TrackDetail>(`/tracks/${encodeURIComponent(trackId)}`),

  getWaveform: (trackId: string, level: number | 'auto' = 'auto', source: StemKind = 'mix') => {
    const params = new URLSearchParams({ level: String(level), source });
    return request<WaveformPeaks>(`/tracks/${encodeURIComponent(trackId)}/waveform?${params}`);
  },

  getHitPoints: (trackId: string) => request<HitPoint[]>(`/tracks/${encodeURIComponent(trackId)}/hit-points`),

  getCandidateEvents: (trackId: string) =>
    request<CandidateEvent[]>(`/tracks/${encodeURIComponent(trackId)}/candidate-events`),

  getVocalLyrics: (trackId: string) =>
    request<VocalLyrics>(`/tracks/${encodeURIComponent(trackId)}/vocal-lyrics`),

  saveVocalLyrics: (trackId: string, text: string, inputFormat: LyricsInputFormat) =>
    request<VocalLyrics>(`/tracks/${encodeURIComponent(trackId)}/vocal-lyrics`, {
      method: 'PUT',
      body: JSON.stringify({ text, inputFormat }),
    }),

  alignVocalLyrics: (trackId: string) =>
    request<VocalLyricsJobResponse>(`/tracks/${encodeURIComponent(trackId)}/vocal-lyrics/align`, {
      method: 'POST',
    }),

  generateVocalLyricsDraft: (trackId: string, inputFormat: LyricsInputFormat = 'japanese') =>
    request<VocalLyricsJobResponse>(`/tracks/${encodeURIComponent(trackId)}/vocal-lyrics/asr-draft`, {
      method: 'POST',
      body: JSON.stringify({ inputFormat }),
    }),

  getVocalLyricsJob: (jobId: string) =>
    request<VocalLyricsJob>(`/vocal-lyrics-jobs/${encodeURIComponent(jobId)}`),

  saveHitPoints: (trackId: string, hitPoints: HitPoint[]) =>
    request<HitPoint[]>(`/tracks/${encodeURIComponent(trackId)}/hit-points`, {
      method: 'PUT',
      body: JSON.stringify({ hitPoints }),
    }),

  saveTempoMap: (trackId: string, tempoMap: TempoSegment[]) =>
    request<TempoSegment[]>(`/tracks/${encodeURIComponent(trackId)}/tempo-map`, {
      method: 'PATCH',
      body: JSON.stringify({ tempoMap }),
    }),

  createHitPoint: (trackId: string, hitPoint: Partial<HitPoint>) =>
    request<HitPoint>(`/tracks/${encodeURIComponent(trackId)}/hit-points`, {
      method: 'POST',
      body: JSON.stringify(hitPoint),
    }),

  updateHitPoint: (trackId: string, hitPointId: string, changes: Partial<HitPoint>) =>
    request<HitPoint>(`/tracks/${encodeURIComponent(trackId)}/hit-points/${encodeURIComponent(hitPointId)}`, {
      method: 'PATCH',
      body: JSON.stringify(changes),
    }),

  deleteHitPoint: (trackId: string, hitPointId: string) =>
    request<void>(`/tracks/${encodeURIComponent(trackId)}/hit-points/${encodeURIComponent(hitPointId)}`, {
      method: 'DELETE',
    }),

  audioUrl: (trackId: string) => apiUrl(`/tracks/${encodeURIComponent(trackId)}/audio`),
  referenceChartAudioUrl: (chartId: string) =>
    apiUrl(`/chart-engine/reference-charts/${encodeURIComponent(chartId)}/audio`),
  chartExportUrl: (trackId: string, generationId?: string) => {
    const params = new URLSearchParams();
    if (generationId) params.set('generationId', generationId);
    const query = params.size ? `?${params}` : '';
    return apiUrl(`/tracks/${encodeURIComponent(trackId)}/chart/export${query}`);
  },
  exportUrl: (
    trackId: string,
    format: 'json' | 'csv' | 'package',
    audio: 'none' | 'reference' | 'full' = 'reference',
  ) => {
    const params = new URLSearchParams({ format });
    if (format === 'package') params.set('audio', audio);
    return apiUrl(`/tracks/${encodeURIComponent(trackId)}/export?${params}`);
  },
};
