export type ProjectStatus = "unprocessed" | "processing" | "completed" | "edited" | "failed";
export type AnalysisMode = "recall" | "balanced" | "clean" | "accurate";
export type HitBand = "low_hit" | "mid_hit" | "high_hit" | "full_band_accent";
export type HitSource = "mix" | "percussive" | "stems" | "fused" | "manual";
export type StemKind = "mix" | "vocals" | "drums" | "bass" | "other";
export type CandidateLane = "vocals" | "melody" | "drums" | "mix";
export type CandidateStatus = "accepted" | "rejected" | "uncertain";

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
  reason: "vocal_presence" | "drum_solo" | "melodic_lead" | "mixed" | "manual";
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
  trackId: string;
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
  /** Audible source and the engine that produced this existing acoustic event. */
  source: string;
  generator: string;
  character: string | null;
  mora: string | null;
  phoneme: string | null;
  eventLevel: string;
  eventPolicy: string | null;
  alignmentUnitId: string | null;
  alignmentUnitIndex: number | null;
  alignmentRunId: string | null;
  characterIndices: number[];
  phonemes: string[];
  /** Raw HuBERT/CTC anchor and its acoustically refined counterpart. */
  alignedSample: number;
  refinedSample: number;
  evidence: Record<string, number>;
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
  subdivision: "1/16" | number;
  subdivisionsPerBeat: number;
  bpm: number;
  beatOffsetSample: number;
  tempoSource: "manual" | "estimated";
  tempoConfidence: number;
  maximumErrorMs: number;
  inputCount: number;
  outputCount: number;
  rejectedOffGrid: number;
  mergedSameGrid: number;
  suppressedNearPreserved?: number;
}

export interface Track {
  id: string;
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

export interface ProjectDetail extends Project {
  track: Track;
}

export type JobStage =
  | "queued"
  | "upload"
  | "decode"
  | "waveform"
  | "tempo"
  | "features"
  | "detect"
  | "refine"
  | "merge"
  | "save"
  | "completed"
  | "failed";

export interface AnalysisJob {
  id: string;
  trackId: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: JobStage;
  progress: number;
  message: string;
  stageTimingsMs: Record<string, number>;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}
