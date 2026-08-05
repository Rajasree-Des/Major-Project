import { cn } from '@/lib/utils'

interface ScenePreviewProps {
  gradient?: string
  label?: string
  className?: string
  aspectRatio?: string
}

export function ScenePreview({
  gradient = 'from-zinc-900 via-neutral-800/80 to-stone-900',
  label,
  className,
  aspectRatio = 'aspect-[16/10]',
}: ScenePreviewProps) {
  return (
    <div className={cn('relative overflow-hidden bg-gradient-to-br', gradient, aspectRatio, className)}>
      <div className="absolute inset-0 opacity-[0.07]">
        <svg className="h-full w-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="preview-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M 24 0 L 0 0 0 24" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#preview-grid)" />
        </svg>
      </div>
      {label && (
        <div className="absolute bottom-3 left-3 rounded-md bg-black/50 backdrop-blur-sm px-2 py-1 text-[10px] font-mono text-white/60 tracking-wide">
          {label}
        </div>
      )}
    </div>
  )
}

const bandGradients: Record<string, string> = {
  thermal: 'from-orange-950 via-red-900/80 to-purple-950',
  blue: 'from-blue-950 via-indigo-900/80 to-slate-950',
  green: 'from-emerald-950 via-green-900/80 to-teal-950',
  red: 'from-rose-950 via-red-900/80 to-orange-950',
}

export function BandPreview({ band, className }: { band: string; className?: string }) {
  return (
    <ScenePreview
      gradient={bandGradients[band] ?? bandGradients.thermal}
      label={band.toUpperCase()}
      aspectRatio="aspect-[4/3]"
      className={cn('rounded-xl border border-border', className)}
    />
  )
}
