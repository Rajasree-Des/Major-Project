import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Play, Download, Trash2, MapPin, Calendar, Cloud } from 'lucide-react'
import { fetchScene } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { BandPreview } from '@/components/shared/ScenePreview'
import { StatusBadge } from '@/components/shared/StatusBadge'
import { FadeIn } from '@/components/shared/PageTransition'
import { formatDate, formatBytes, cn } from '@/lib/utils'
import type { BandType } from '@/types'

const bands: { id: BandType; label: string }[] = [
  { id: 'thermal', label: 'Thermal' },
  { id: 'blue', label: 'Blue' },
  { id: 'green', label: 'Green' },
  { id: 'red', label: 'Red' },
]

const timeline = [
  { step: 'Downloaded', date: '2026-02-28', done: true },
  { step: 'Preprocessed', date: '2026-03-01', done: true },
  { step: 'Inference', date: '2026-03-01', done: true },
  { step: 'Exported', date: '2026-03-01', done: false },
]

export function SceneDetailsPage() {
  const { id } = useParams<{ id: string }>()
  const [activeBand, setActiveBand] = useState<BandType>('thermal')
  const { data: scene, loading } = useAsyncData(() => fetchScene(id!), [id])

  if (loading) {
    return <div className="rounded-2xl border border-border bg-card aspect-video animate-pulse" />
  }

  if (!scene) {
    return (
      <div className="text-center py-20">
        <p className="text-muted">Scene not found</p>
        <Link to="/datasets"><Button variant="secondary" className="mt-4">Back to Datasets</Button></Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <FadeIn>
        <div className="flex items-center gap-4">
          <Link to="/datasets">
            <Button variant="ghost" size="icon"><ArrowLeft className="h-4 w-4" /></Button>
          </Link>
          <div>
            <h2 className="text-lg font-semibold">{scene.location}</h2>
            <p className="font-mono text-xs text-muted-dim mt-0.5">{scene.sceneId}</p>
          </div>
        </div>
      </FadeIn>

      <div className="grid gap-8 lg:grid-cols-5">
        <FadeIn delay={0.05} className="lg:col-span-3 space-y-4">
          <BandPreview band={activeBand} className="aspect-[16/10]" />

          <div className="flex gap-2">
            {bands.map((b) => (
              <button
                key={b.id}
                onClick={() => setActiveBand(b.id)}
                className={cn(
                  'px-4 py-2 rounded-xl text-xs font-medium transition-all',
                  activeBand === b.id
                    ? 'bg-primary/10 text-primary border border-primary/20'
                    : 'bg-surface text-muted border border-border hover:text-foreground'
                )}
              >
                {b.label}
              </button>
            ))}
          </div>

          <div className="rounded-xl border border-border bg-card p-6 space-y-3">
            <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider">Metadata</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div><span className="text-muted">Path/Row</span><p className="font-mono text-xs mt-0.5">{scene.pathRow}</p></div>
              <div><span className="text-muted">CRS</span><p className="font-mono text-xs mt-0.5">EPSG:32643</p></div>
              <div><span className="text-muted">Thermal Band</span><p className="mt-0.5">{scene.thermalBand.name} · {scene.thermalBand.resolution}</p></div>
              <div><span className="text-muted">RGB Bands</span><p className="mt-0.5">{scene.rgbBands.map(b => b.name).join(', ')}</p></div>
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.1} className="lg:col-span-2 space-y-4">
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <StatusBadge status={scene.status} />
              <Badge variant={scene.sentinelRgbAvailable ? 'success' : 'secondary'}>
                Sentinel {scene.sentinelRgbAvailable ? 'Available' : 'N/A'}
              </Badge>
            </div>

            <div className="space-y-3 text-sm">
              {[
                { icon: Calendar, label: 'Acquisition', value: formatDate(scene.acquisitionDate) },
                { icon: MapPin, label: 'Coordinates', value: `${scene.centerLat.toFixed(4)}°N, ${scene.centerLon.toFixed(4)}°E` },
                { icon: Cloud, label: 'Cloud Cover', value: `${scene.cloudCover}%` },
              ].map(({ icon: Icon, label, value }) => (
                <div key={label} className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted"><Icon className="h-3.5 w-3.5" />{label}</span>
                  <span className="text-foreground">{value}</span>
                </div>
              ))}
              <div className="flex items-center justify-between">
                <span className="text-muted">Resolution</span>
                <span>Thermal 100m · RGB 30m</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted">File Size</span>
                <span>{formatBytes(scene.downloadSizeBytes)}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-2"><span className="text-muted">Processing</span><span>{scene.processingProgress}%</span></div>
              <Progress value={scene.processingProgress} />
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-6">
            <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider mb-4">Processing Timeline</h3>
            <div className="space-y-3">
              {timeline.map((t) => (
                <div key={t.step} className="flex items-center gap-3">
                  <div className={cn('h-2 w-2 rounded-full', t.done ? 'bg-success' : 'bg-border')} />
                  <span className="text-sm flex-1">{t.step}</span>
                  <span className="text-xs text-muted-dim">{t.date}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <Link to="/preprocessing"><Button className="w-full"><Play className="h-4 w-4" /> Preprocess</Button></Link>
            <Button variant="secondary" className="w-full"><Download className="h-4 w-4" /> Export</Button>
            <Button variant="ghost" className="w-full text-error hover:text-error hover:bg-error/10"><Trash2 className="h-4 w-4" /> Delete</Button>
          </div>
        </FadeIn>
      </div>
    </div>
  )
}
