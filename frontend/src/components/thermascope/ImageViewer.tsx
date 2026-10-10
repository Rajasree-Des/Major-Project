import { useRef, useState } from 'react'
import { Download, Loader2, ZoomIn, ZoomOut } from 'lucide-react'
import { cn } from '@/lib/utils'

type Props = {
  src: string
  alt: string
  className?: string
  downloadName?: string
}

export function ImageViewer({ src, alt, className, downloadName }: Props) {
  const [scale, setScale] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const pan = useRef({ x: 0, y: 0, dragging: false, sx: 0, sy: 0 })
  const [offset, setOffset] = useState({ x: 0, y: 0 })

  return (
    <div className={cn('relative flex flex-col rounded-xl border border-border-slate bg-surface shadow-glow overflow-hidden', className)}>
      <div className="flex items-center justify-end gap-1 border-b border-border-slate px-2 py-1.5 bg-surface-elevated">
        <button type="button" className="p-1.5 text-muted hover:text-accent" onClick={() => setScale((s) => Math.min(4, s + 0.25))} aria-label="Zoom in">
          <ZoomIn className="h-4 w-4" />
        </button>
        <button type="button" className="p-1.5 text-muted hover:text-accent" onClick={() => setScale((s) => Math.max(0.5, s - 0.25))} aria-label="Zoom out">
          <ZoomOut className="h-4 w-4" />
        </button>
        {downloadName && (
          <a href={src} download={downloadName} className="p-1.5 text-muted hover:text-accent" aria-label="Download">
            <Download className="h-4 w-4" />
          </a>
        )}
      </div>
      <div
        className="relative flex min-h-[280px] max-h-[520px] items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing bg-midnight/50"
        onMouseDown={(e) => {
          pan.current = { ...pan.current, dragging: true, sx: e.clientX - offset.x, sy: e.clientY - offset.y }
        }}
        onMouseMove={(e) => {
          if (!pan.current.dragging) return
          setOffset({ x: e.clientX - pan.current.sx, y: e.clientY - pan.current.sy })
        }}
        onMouseUp={() => {
          pan.current.dragging = false
        }}
        onMouseLeave={() => {
          pan.current.dragging = false
        }}
      >
        {loading && !error && <Loader2 className="absolute h-8 w-8 animate-spin text-accent" />}
        {error ? (
          <p className="text-sm text-muted">Failed to load image</p>
        ) : (
          <img
            src={src}
            alt={alt}
            draggable={false}
            className="max-h-[480px] w-auto object-contain select-none transition-transform duration-100"
            style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }}
            onLoad={() => setLoading(false)}
            onError={() => {
              setLoading(false)
              setError(true)
            }}
          />
        )}
      </div>
    </div>
  )
}
