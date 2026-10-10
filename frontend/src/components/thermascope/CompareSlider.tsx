import { useState } from 'react'

type Props = {
  beforeSrc: string
  afterSrc: string
  beforeLabel?: string
  afterLabel?: string
}

export function CompareSlider({ beforeSrc, afterSrc, beforeLabel = 'Before', afterLabel = 'After' }: Props) {
  const [pos, setPos] = useState(50)

  return (
    <div className="relative aspect-[4/3] max-h-[420px] w-full overflow-hidden rounded-xl border border-border-slate bg-midnight select-none">
      <img src={afterSrc} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" />
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <img src={beforeSrc} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" style={{ maxWidth: 'none', width: `${100 / (pos / 100)}%` }} />
      </div>
      <div className="absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_8px_rgba(34,211,238,0.8)]" style={{ left: `${pos}%` }} />
      <input
        type="range"
        min={5}
        max={95}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        className="absolute inset-x-4 bottom-3 z-10 w-[calc(100%-2rem)] accent-accent opacity-90"
        aria-label="Compare slider"
      />
      <span className="absolute left-3 top-3 rounded bg-black/70 px-2 py-0.5 text-[10px] text-text">{beforeLabel}</span>
      <span className="absolute right-3 top-3 rounded bg-black/70 px-2 py-0.5 text-[10px] text-text">{afterLabel}</span>
    </div>
  )
}
