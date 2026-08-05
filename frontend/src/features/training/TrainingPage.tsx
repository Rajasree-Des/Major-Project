import { fetchTrainingMetrics } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Card } from '@/components/ui/card'
import { Progress, CircularProgress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { FadeIn } from '@/components/shared/PageTransition'

export function TrainingPage() {
  const { data: m, loading } = useAsyncData(() => fetchTrainingMetrics())

  if (loading || !m) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-40 rounded-xl border border-border animate-pulse" />
        ))}
      </div>
    )
  }

  const epochPct = (m.epoch / m.totalEpochs) * 100
  const stepPct = (m.currentStep / m.stepsPerEpoch) * 100
  const samplePct = (m.samplesProcessed / m.totalSamples) * 100
  const gpuMemPct = (m.gpuMemoryUsed / m.gpuMemoryTotal) * 100

  return (
    <div className="space-y-8">
      <FadeIn>
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Training Monitor</h2>
            <p className="text-sm text-muted mt-1">{m.datasetName}</p>
          </div>
          <Badge variant="warning">Running</Badge>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { label: 'Epoch', value: `${m.epoch} / ${m.totalEpochs}`, pct: epochPct },
            { label: 'Train Loss', value: m.trainLoss.toFixed(4), sub: `Val: ${m.valLoss.toFixed(4)}` },
            { label: 'Learning Rate', value: m.learningRate.toExponential(1), sub: 'Cosine annealing' },
            { label: 'ETA', value: m.eta, sub: 'Estimated remaining' },
          ].map((item) => (
            <Card key={item.label} className="p-6">
              <p className="text-[11px] font-medium text-muted-dim uppercase tracking-wider">{item.label}</p>
              <p className="text-2xl font-semibold mt-2 font-mono">{item.value}</p>
              {'sub' in item && item.sub && <p className="text-xs text-muted mt-1">{item.sub}</p>}
              {'pct' in item && item.pct !== undefined && (
                <Progress value={item.pct} className="mt-3" />
              )}
            </Card>
          ))}
        </div>
      </FadeIn>

      <div className="grid gap-6 lg:grid-cols-3">
        <FadeIn delay={0.1} className="lg:col-span-2">
          <Card className="p-8">
            <div className="flex items-center gap-8">
              <div className="relative">
                <CircularProgress value={epochPct} size={120} strokeWidth={4} />
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-2xl font-semibold">{Math.round(epochPct)}%</span>
                  <span className="text-[10px] text-muted-dim">Epoch</span>
                </div>
              </div>
              <div className="flex-1 space-y-5">
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-muted">Step Progress</span>
                    <span className="font-mono">{m.currentStep}/{m.stepsPerEpoch}</span>
                  </div>
                  <Progress value={stepPct} />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-muted">Samples</span>
                    <span className="font-mono">{m.samplesProcessed.toLocaleString()}/{m.totalSamples.toLocaleString()}</span>
                  </div>
                  <Progress value={samplePct} indicatorClassName="bg-success" />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-muted">GPU Memory</span>
                    <span className="font-mono">{m.gpuMemoryUsed}/{m.gpuMemoryTotal} GB</span>
                  </div>
                  <Progress value={gpuMemPct} indicatorClassName="bg-warning" />
                </div>
              </div>
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={0.15}>
          <Card className="p-6 space-y-5">
            <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider">Configuration</h3>
            {[
              { label: 'Model Version', value: m.modelVersion },
              { label: 'Checkpoint', value: m.checkpoint },
              { label: 'Batch Size', value: m.batchSize },
              { label: 'Best Val Loss', value: m.bestValLoss.toFixed(4) },
              { label: 'GPU Usage', value: `${m.gpuUsage}%` },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between text-sm gap-4">
                <span className="text-muted shrink-0">{label}</span>
                <span className="font-mono text-xs text-right truncate">{value}</span>
              </div>
            ))}
          </Card>
        </FadeIn>
      </div>
    </div>
  )
}
