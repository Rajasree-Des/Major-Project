import { cn } from '@/lib/utils'

interface MapPlaceholderProps {
  lat: number
  lon: number
  className?: string
}

export function MapPlaceholder({ lat, lon, className }: MapPlaceholderProps) {
  return (
    <div className={cn('relative overflow-hidden rounded-lg border border-border bg-[#0d1a2d]', className)}>
      <div className="absolute inset-0 opacity-30">
        <svg className="h-full w-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="map-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#2563EB" strokeWidth="0.3" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#map-grid)" />
        </svg>
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          <div className="absolute -inset-8 rounded-full border border-accent/20 animate-pulse" />
          <div className="absolute -inset-4 rounded-full border border-accent/40" />
          <div className="h-3 w-3 rounded-full bg-accent shadow-lg shadow-accent/50" />
        </div>
      </div>
      <div className="absolute bottom-3 left-3 rounded bg-black/60 px-2 py-1 font-mono text-xs text-white/70">
        {lat.toFixed(4)}°N, {lon.toFixed(4)}°E
      </div>
      <div className="absolute top-3 right-3 rounded bg-black/60 px-2 py-1 text-xs text-white/50">
        Map View
      </div>
    </div>
  )
}
