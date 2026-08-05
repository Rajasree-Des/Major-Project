import { useState, useRef, useCallback } from 'react'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import { fetchEvaluation } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Card } from '@/components/ui/card'
import { Select } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { FadeIn } from '@/components/shared/PageTransition'
import type { MetricExplanation } from '@/types'

function ComparisonViewer() {
  const [position, setPosition] = useState(50)
  const [zoom, setZoom] = useState(1)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const updatePosition = useCallback((clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    setPosition(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }, [])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} className="h-8 w-8 rounded-lg border border-border flex items-center justify-center text-muted hover:text-foreground transition-colors">
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setZoom((z) => Math.min(3, z + 0.25))} className="h-8 w-8 rounded-lg border border-border flex items-center justify-center text-muted hover:text-foreground transition-colors">
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setZoom(1)} className="h-8 w-8 rounded-lg border border-border flex items-center justify-center text-muted hover:text-foreground transition-colors">
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </div>
        <span className="text-xs text-muted-dim font-mono">{Math.round(zoom * 100)}%</span>
      </div>

      <div
        ref={containerRef}
        className="relative aspect-[16/10] overflow-hidden rounded-xl border border-border cursor-col-resize select-none"
        style={{ transform: `scale(${zoom})`, transformOrigin: 'center' }}
        onPointerMove={(e) => dragging.current && updatePosition(e.clientX)}
        onPointerUp={() => { dragging.current = false }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-orange-950 via-red-900/80 to-purple-950">
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-white/30 font-mono">After — 30m Enhanced</span>
          </div>
        </div>
        <div
          className="absolute inset-0 bg-gradient-to-br from-zinc-900 via-neutral-800/80 to-stone-900"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-white/30 font-mono">Before — 100m ST_B10</span>
          </div>
        </div>
        <div className="absolute top-0 bottom-0 w-px bg-white/60 z-10" style={{ left: `${position}%` }}>
          <div
            className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-lg flex items-center justify-center"
            onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); updatePosition(e.clientX) }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12"><path d="M3 2L1 6l2 4M9 2l2 4-2 4" stroke="#09090b" strokeWidth="1.2" fill="none" strokeLinecap="round" /></svg>
          </div>
        </div>
        <div className="absolute top-3 left-3 rounded-md bg-black/50 px-2 py-1 text-[10px] text-white/70">Before</div>
        <div className="absolute top-3 right-3 rounded-md bg-black/50 px-2 py-1 text-[10px] text-white/70">After</div>
      </div>
    </div>
  )
}

export function EvaluationPage() {
  const { data: metricsList, loading } = useAsyncData(() => fetchEvaluation())
  const [selectedId, setSelectedId] = useState('')
  const metrics = metricsList?.find((m) => m.sceneId === (selectedId || metricsList[0]?.sceneId)) ?? metricsList?.[0]

  const explanations: MetricExplanation[] = metrics ? [
    { name: 'PSNR', value: metrics.psnr, unit: 'dB', description: 'Peak Signal-to-Noise Ratio measures reconstruction fidelity. Higher values indicate less distortion between predicted and reference imagery.', threshold: '> 30 dB', pass: metrics.psnr > 30 },
    { name: 'SSIM', value: metrics.ssim, unit: '', description: 'Structural Similarity Index captures luminance, contrast, and structural correlation between image patches.', threshold: '> 0.90', pass: metrics.ssim > 0.9 },
    { name: 'LPIPS', value: metrics.lpips, unit: '', description: 'Learned Perceptual Image Patch Similarity uses deep features to assess perceptual distance. Lower is better.', threshold: '< 0.10', pass: metrics.lpips < 0.1 },
    { name: 'RMSE', value: metrics.rmse, unit: '°C', description: 'Root Mean Square Error of brightness temperature reconstruction in degrees Celsius.', threshold: '< 5.0°C', pass: metrics.rmse < 5 },
  ] : []

  if (loading) return <div className="h-96 rounded-2xl border border-border animate-pulse" />

  return (
    <div className="space-y-8">
      <FadeIn>
        <div className="flex items-end justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Evaluation</h2>
            <p className="text-sm text-muted mt-1">Quantitative assessment of super-resolution and colorization quality</p>
          </div>
          {metricsList && (
            <Select value={selectedId || metricsList[0]?.sceneId} onChange={(e) => setSelectedId(e.target.value)} className="w-72">
              {metricsList.map((m) => (
                <option key={m.sceneId} value={m.sceneId}>{m.sceneId.slice(0, 32)}…</option>
              ))}
            </Select>
          )}
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <Card className="p-6">
          <ComparisonViewer />
        </Card>
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {explanations.map((m) => (
            <Card key={m.name} className="p-6">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[11px] font-medium text-muted-dim uppercase tracking-wider">{m.name}</p>
                <Badge variant={m.pass ? 'success' : 'warning'}>{m.pass ? 'Pass' : 'Review'}</Badge>
              </div>
              <p className="text-3xl font-semibold font-mono">
                {m.value}{m.unit && <span className="text-base text-muted ml-1">{m.unit}</span>}
              </p>
              <p className="text-[11px] text-muted-dim mt-1">Target: {m.threshold}</p>
            </Card>
          ))}
        </div>
      </FadeIn>

      <FadeIn delay={0.15}>
        <Card className="p-6">
          <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider mb-4">Metric Explanations</h3>
          <div className="space-y-4">
            {explanations.map((m) => (
              <div key={m.name} className="rounded-xl bg-surface border border-border p-4">
                <p className="text-sm font-medium mb-1">{m.name}</p>
                <p className="text-xs text-muted leading-relaxed">{m.description}</p>
              </div>
            ))}
          </div>
        </Card>
      </FadeIn>
    </div>
  )
}
