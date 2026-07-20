export function clampSample(sample: number, sampleCount: number): number {
  if (!Number.isFinite(sample)) return 0;
  return Math.max(0, Math.min(Math.max(0, sampleCount - 1), Math.round(sample)));
}

export function sampleToSeconds(sample: number, sampleRate: number): number {
  if (sampleRate <= 0) return 0;
  return Math.round(sample) / sampleRate;
}

export function secondsToSample(seconds: number, sampleRate: number, sampleCount = Number.MAX_SAFE_INTEGER): number {
  return clampSample(Math.round(seconds * sampleRate), sampleCount);
}

export function millisecondsToSamples(milliseconds: number, sampleRate: number): number {
  return Math.round((milliseconds / 1000) * sampleRate);
}

export function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const totalMilliseconds = Math.round(safeSeconds * 1000);
  const minutes = Math.floor(totalMilliseconds / 60_000);
  const secs = Math.floor((totalMilliseconds % 60_000) / 1000);
  const millis = totalMilliseconds % 1000;
  return `${minutes}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export function formatMusicalPosition(
  sample: number,
  sampleRate: number,
  bpm: number,
  beatOffsetSample: number,
  numerator: number,
  subdivisionsPerBeat = 4,
): string {
  if (bpm <= 0 || sampleRate <= 0) return '1:1:1';
  const samplesPerBeat = (sampleRate * 60) / bpm;
  const relativeBeats = (sample - beatOffsetSample) / samplesPerBeat;
  const absoluteSubdivision = Math.max(0, Math.round(relativeBeats * subdivisionsPerBeat));
  const beatIndex = Math.floor(absoluteSubdivision / subdivisionsPerBeat);
  const subdivision = (absoluteSubdivision % subdivisionsPerBeat) + 1;
  const bar = Math.floor(beatIndex / numerator) + 1;
  const beat = (beatIndex % numerator) + 1;
  return `${bar}:${beat}:${subdivision}`;
}
