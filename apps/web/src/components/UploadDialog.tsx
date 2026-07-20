import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import type { AnalysisMode } from '../types';
import { formatBytes } from '../utils/time';

const MAX_FILE_SIZE = 250 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ['wav', 'flac', 'mp3', 'm4a', 'aac', 'ogg'];

export function UploadDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [mode, setMode] = useState<AnalysisMode>('balanced');
  const [sensitivity, setSensitivity] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) {
      setFile(null);
      setTitle('');
      setBusy(false);
      setError('');
    }
  }, [open]);

  if (!open) return null;

  const chooseFile = (nextFile?: File) => {
    if (!nextFile) return;
    const extension = nextFile.name.split('.').pop()?.toLowerCase() ?? '';
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setError(`不支持 .${extension || '未知'} 格式，请选择 WAV、FLAC、MP3、M4A、AAC 或 OGG。`);
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError('文件超过 250 MB 限制。');
      return;
    }
    setError('');
    setFile(nextFile);
    setTitle(nextFile.name.replace(/\.[^.]+$/, ''));
  };

  const submit = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError('');
    try {
      const uploaded = await api.uploadTrack(file, { title: title.trim() || undefined });
      const job = await api.analyzeTrack(uploaded.track.id, mode, sensitivity);
      navigate(`/projects/${uploaded.project.id}?job=${encodeURIComponent(job.jobId)}`);
    } catch (reason) {
      setBusy(false);
      setError(reason instanceof ApiError ? reason.message : '导入失败，请检查后端和 ffmpeg 后重试。');
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
      <section className="upload-dialog" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <button className="icon-button dialog-close" onClick={onClose} disabled={busy} aria-label="关闭">×</button>
        <div className="eyebrow">NEW ANALYSIS</div>
        <h2 id="upload-title">导入音频</h2>
        <p className="muted">原始文件会被保留；检测位置始终映射回原始采样点。</p>
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          accept=".wav,.flac,.mp3,.m4a,.aac,.ogg,audio/*"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />
        <button className={`file-drop${file ? ' has-file' : ''}`} onClick={() => inputRef.current?.click()} type="button">
          <span className="upload-glyph">↥</span>
          {file ? (
            <span className="file-summary">
              <strong>{file.name}</strong>
              <small>{file.type || file.name.split('.').pop()?.toUpperCase()} · {formatBytes(file.size)}</small>
            </span>
          ) : (
            <span className="file-summary">
              <strong>选择本地音频</strong>
              <small>WAV / FLAC / MP3 / M4A / AAC / OGG，最大 250 MB</small>
            </span>
          )}
        </button>
        <label className="field-label">歌曲名称<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="默认使用文件名" /></label>
        <div className="dialog-row">
          <label className="field-label">分析模式
            <select value={mode} onChange={(event) => setMode(event.target.value as AnalysisMode)}>
              <option value="recall">高召回</option><option value="balanced">平衡</option><option value="clean">干净</option><option value="accurate">精确</option>
            </select>
          </label>
          <label className="field-label">灵敏度 <span>{Math.round(sensitivity * 100)}%</span>
            <input type="range" min="0" max="1" step="0.05" value={sensitivity} onChange={(event) => setSensitivity(Number(event.target.value))} />
          </label>
        </div>
        {mode === 'accurate' ? <p className="inline-note">若本机没有 Demucs，任务会自动回退到平衡模式并给出提示。</p> : null}
        {error ? <div className="error-banner" role="alert">{error}</div> : null}
        <button className="primary-button full-button" disabled={!file || busy} onClick={submit}>
          {busy ? <><span className="spinner" /> 正在上传…</> : '生成击打点'}
        </button>
      </section>
    </div>
  );
}
