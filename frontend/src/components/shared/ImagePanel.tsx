import { cn } from '@/lib/utils'
import { Loader2, ImageOff } from 'lucide-react'
import { useState } from 'react'

type Props = {
  src: string
  alt: string
  label?: string
  className?: string
  caption?: string
}

export function ImagePanel({ src, alt, label, className, caption }: Props) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  return (
    <div className={cn('rounded-2xl border border-workspace-border bg-white shadow-sm overflow-hidden', className)}>
      {label && (
        <div className="border-b border-workspace-border px-4 py-2.5 flex items-center justify-between bg-workspace-muted/40">
          <span className="text-xs font-semibold uppercase tracking-wider text-workspace-muted-fg">{label}</span>
        </div>
      )}
      <div className="relative flex min-h-[220px] items-center justify-center bg-slate-950/5 p-3">
        {loading && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
            <Loader2 className="h-8 w-8 animate-spin text-teal-600" />
          </div>
        )}
        {error ? (
          <div className="flex flex-col items-center gap-2 text-workspace-muted-fg py-12">
            <ImageOff className="h-10 w-10 opacity-50" />
            <p className="text-sm">Image unavailable</p>
          </div>
        ) : (
          <img
            src={src}
            alt={alt}
            className="max-h-[420px] w-full object-contain"
            onLoad={() => setLoading(false)}
            onError={() => {
              setLoading(false)
              setError(true)
            }}
          />
        )}
      </div>
      {caption && <p className="px-4 py-2 text-xs text-workspace-muted-fg border-t border-workspace-border">{caption}</p>}
    </div>
  )
}
