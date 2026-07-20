export type ProjectStatus = 'unprocessed' | 'processing' | 'completed' | 'edited' | 'failed';
export type AnalysisMode = 'recall' | 'balanced' | 'clean' | 'accurate';
export type HitBand = 'low_hit' | 'mid_hit' | 'high_hit' | 'full_band_accent';
export type HitSource = 'mix' | 'percussive' | 'stems' | 'fused' | 'manual';
export type StemKind = 'mix' | 'vocals' | 'drums' | 'bass' | 'other';
export type CandidateLane = 'vocals' | 'melody' | 'drums' | 'mix';
export type CandidateStatus = 'accepted' | 'rejected' | 'uncertain';
export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';
export type LyricsInputFormat = 'japanese' | 'kana' | 'romaji' | 'lrc';
export type VocalLyricsStatus = 'empty' | 'draft' | 'saved' | 'queued' | 'processing' | 'completed' | 'failed';
export type VocalLyricsStage =
  | 'idle'
  | 'queued'
  | 'separating_vocals'
  | 'detecting_vocal_activity'
  | 'transcribing'
  | 'normalizing_pronunciation'
  | 'aligning_lyrics'
  | 'refining_samples'
  | 'saving_results'
  | 'completed';
export type AlignmentMethodId = 'qwen' | 'mfa' | 'ctc' | 'singing' | 'hybrid';
export type AlignmentLayer = 'character' | 'mora' | 'phoneme';
export type AlignmentResultStatus =
  | 'empty'
  | 'queued'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'unavailable';

export interface AlignmentMethod {
  id: AlignmentMethodId;
  name: string;
  available: boolean;
  reason?: string | null;
  description?: string | null;
}

export interface AlignmentToken {
  id: string;
  text: string;
  phoneme: string | null;
  startSample: number;
  endSample: number;
  confidence: number;
  method: AlignmentMethodId;
}

export interface AlignmentAcousticEvidence {
  energy: number;
  spectralChange: number;
  pitchChange: number;
}

/**
 * A timestamped unit emitted by the Japanese HuBERT hierarchy. All positions
 * stay in the track's original sample domain. The aligned fields are the raw
 * CTC result; refined fields are the acoustically refined result.
 */
export interface AlignmentHierarchyUnit {
  id: string;
  index: number;
  level: AlignmentLayer;
  text: string;
  kana: string | null;
  mora: string | null;
  phoneme: string | null;
  kind: string | null;
  characterIndices: number[];
  moraIndices: number[];
  phonemeIndices: number[];
  alignedStartSample: number;
  alignedEndSample: number;
  refinedStartSample: number;
  refinedEndSample: number;
  alignedSample: number;
  refinedSample: number;
  confidence: number;
  observedTokenIndex: number | null;
  matchOperation: string | null;
  evidence: AlignmentAcousticEvidence | null;
}

export interface AlignmentHierarchy {
  phonemes: AlignmentHierarchyUnit[];
  moras: AlignmentHierarchyUnit[];
  characters: AlignmentHierarchyUnit[];
}

export interface AlignmentError {
  code: string;
  message: string;
  details?: unknown;
}

export interface AlignmentResult {
  runId: string;
  trackId: string;
  method: AlignmentMethodId;
  status: AlignmentResultStatus;
  sampleRate: number;
  sampleCount: number;
  tokens: AlignmentToken[];
  hierarchy?: AlignmentHierarchy | null;
  warnings: string[];
  error: AlignmentError | string | null;
  metadata: Record<string, unknown>;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface AlignmentReport {
  runId: string;
  trackId: string;
  method: AlignmentMethodId;
  score: number;
  coverage: number;
  acoustic: number;
  rhythm: number;
  stability: number;
  lyricTokenCount: number;
  alignedTokenCount: number;
  details: Record<string, unknown>;
  createdAt: string | null;
}

export interface StemDescriptor {
  source: StemKind;
  available: boolean;
  waveformUrl: string;
  audioUrl?: string | null;
}

export interface FocusSegment {
  id: string;
  startSample: number;
  endSample: number;
  focusSource: StemKind;
  confidence: number;
  reason: 'vocal_presence' | 'drum_solo' | 'melodic_lead' | 'mixed' | 'manual';
  manuallyEdited: boolean;
  evidence?: Partial<Record<StemKind, number>>;
  alternatives?: { source: StemKind; score: number }[];
}

export interface Project {
  id: string;
  title: string;
  artist: string;
  genre: string;
  coverUrl: string;
  status: ProjectStatus;
  createdAt: string;
  updatedAt: string;
  trackId: string | null;
}

export interface ProjectSummary extends Project {
  bpm?: number | null;
  durationSec?: number | null;
  hitPointCount?: number | null;
  analysisMode?: AnalysisMode | null;
  track?: TrackDetail | null;
}

export interface TempoSegment {
  id: string;
  startSample: number;
  bpm: number;
  timeSignatureNumerator: number;
  timeSignatureDenominator: number;
  beatOffsetSample: number;
  confidence: number;
  manuallyEdited: boolean;
}

export interface HitPoint {
  id: string;
  sample: number;
  timeSec: number;
  acousticSample: number;
  chartSample: number;
  detectedSample: number;
  refinedSample: number;
  snappedSample: number;
  snapErrorMs: number;
  band: HitBand;
  confidence: number;
  salience: number;
  source: HitSource;
  primaryStem: StemKind;
  stemEvidence: Partial<Record<StemKind, number>>;
  detectorVotes: string[];
  manuallyEdited: boolean;
  locked: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface CandidateEvent {
  id: string;
  sample: number;
  timeSec: number;
  acousticSample: number;
  chartSample: number;
  snapErrorMs: number;
  lane: CandidateLane;
  sourceEvidence: Record<string, number>;
  semanticEvidence: Record<string, number>;
  confidence: number;
  status: CandidateStatus;
  gridType: string;
  gridConfidence: number;
  source?: 'vocals' | string | null;
  generator?: string | null;
  character?: string | null;
  mora?: string | null;
  phoneme?: string | null;
  eventLevel?: string;
  eventPolicy?: string | null;
  alignmentUnitId?: string | null;
  alignmentUnitIndex?: number | null;
  alignmentRunId?: string | null;
  characterIndices?: number[];
  phonemes?: string[];
  alignedSample?: number | null;
  refinedSample?: number | null;
  evidence?: {
    hubert: number;
    energy: number;
    pitch: number;
    rhythm: number;
    spectralChange?: number;
  } | null;
  hitPointId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface AnalysisMetadata {
  version: string;
  mode: AnalysisMode;
  parameters: Record<string, unknown>;
  elapsedMs: number;
  bpmConfidence: number;
  warnings: string[];
  createdAt: string;
  rhythmConstraint?: RhythmConstraintMetadata;
}

export interface RhythmConstraintMetadata {
  applied: boolean;
  subdivision: '1/16' | number;
  subdivisionsPerBeat: number;
  bpm: number;
  beatOffsetSample: number;
  tempoSource: 'manual' | 'estimated';
  tempoConfidence: number;
  maximumErrorMs: number;
  inputCount: number;
  outputCount: number;
  rejectedOffGrid: number;
  mergedSameGrid: number;
  suppressedNearPreserved?: number;
}

export interface TrackDetail {
  id: string;
  projectId: string;
  createdAt: string;
  updatedAt: string;
  originalFileName: string;
  audioUrl: string;
  format: string;
  originalSampleRate: number;
  channels: number;
  sampleCount: number;
  durationSec: number;
  leadingSilenceSamples: number;
  analysis: AnalysisMetadata | null;
  tempoMap: TempoSegment[];
  hitPoints: HitPoint[];
  candidateEvents: CandidateEvent[];
  stems: StemDescriptor[];
  focusMap: FocusSegment[];
  waveformUrl: string;
}

export interface ProjectDetail extends Project {
  track: TrackDetail | null;
}

export interface WaveformPeaks {
  trackId: string;
  source: StemKind;
  sampleRate: number;
  sampleCount: number;
  level: number;
  windowSize: number;
  mins: number[];
  maxs: number[];
}

export interface AnalysisJob {
  id: string;
  trackId: string;
  status: JobStatus;
  stage: string;
  progress: number;
  stageTimings: Record<string, number>;
  error: { code: string; message: string; details?: unknown } | string | null;
  warnings: string[];
  createdAt: string;
  updatedAt: string;
}

export interface VocalLyricsAnchor {
  id: string;
  index: number;
  originalText: string;
  kana: string;
  romaji: string;
  wordStart: boolean;
  active: boolean;
  chartCandidate: boolean;
  activityScore: number;
  attackScore: number;
  pitchScore?: number;
  transitionScore?: number;
  acousticConfidence?: number;
  semanticUnit?: 'phrase';
  alignmentShiftMs: number;
  chunkMatchConfidence?: number;
  alignedSample: number;
  refinedSample: number;
  gridSample: number | null;
  confidence: number;
}

export interface VocalLyrics {
  trackId: string;
  text: string;
  inputFormat: LyricsInputFormat;
  status: VocalLyricsStatus;
  stage: VocalLyricsStage;
  progress: number;
  anchors: VocalLyricsAnchor[];
  error: { code: string; message: string; details?: unknown } | null;
  updatedAt: string | null;
}

export interface VocalLyricsJob {
  id: string;
  trackId: string;
  kind: 'alignment' | 'asr_draft';
  status: JobStatus;
  stage: VocalLyricsStage;
  progress: number;
  stageTimings: Record<string, number>;
  error: { code: string; message: string; details?: unknown } | null;
  result: VocalLyrics | null;
  createdAt: string;
  updatedAt: string;
}

export interface VocalLyricsJobResponse {
  jobId: string;
  status: JobStatus;
}

export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown };
}

export interface ProjectListResponse {
  items: ProjectSummary[];
  total: number;
}

export interface UploadResponse {
  project: Project;
  track: TrackDetail;
}

export interface AnalyzeResponse {
  jobId: string;
  status: JobStatus;
}

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export interface HitDisplayFilters {
  band: 'all' | HitBand | 'manual';
  stem: 'all' | StemKind;
  minConfidence: number;
  onlyUnedited: boolean;
  onlyLowConfidence: boolean;
  onlyOffGrid: boolean;
  showGrid: boolean;
  showHitPoints: boolean;
  showWaveform: boolean;
  showCandidateEvents: boolean;
  candidateLane: 'all' | CandidateLane;
}

export type GridSubdivision = '1/4' | '1/8' | '1/12' | '1/16' | '1/24' | '1/32';
