import { Link } from 'react-router-dom';

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link className="brand" to="/" aria-label="BeatForge Studio 首页">
      <span className="brand-mark" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </span>
      {!compact && (
        <span className="brand-copy">
          <strong>BeatForge</strong>
          <small>STUDIO · 卡点工坊</small>
        </span>
      )}
    </Link>
  );
}
