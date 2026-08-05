import { Link } from 'react-router-dom'
import { ArrowRight, Circle, CheckCircle2, Clock, Loader2 } from 'lucide-react'
import { fetchDashboard } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { StatSkeleton } from '@/components/ui/skeleton'
import { FadeIn } from '@/components/shared/PageTransition'
import { formatBytes, formatDateTime, cn } from '@/lib/utils'
import type { ProcessingStepStatus } from '@/types'

const stageIcons: Record<ProcessingStepStatus, typeof CheckCircle2> = {
  completed: CheckCircle2,
  running: Loader2,
  waiting: Clock,
  pending: Clock,
  failed: Circle,
}

export function DashboardPage() {
  const { data, loading } = useAsyncData(() => fetchDashboard())

  if (loading || !data) {
    return (
      <div className="space-y-8">
        <div className="rounded-2xl border border-border bg-card p-10 h-48 animate-pulse" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <StatSkeleton key={i} />)}</div>
      </div>
    )
  }

  const { stats, activities, health } = data
  const storagePct = (stats.storageUsedBytes / stats.storageTotalBytes) * 100

  return (
    <div className="space-y-8">
      <FadeIn>
        <Card className="relative overflow-hidden border-border-subtle/50">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent" />
          <CardContent className="relative p-10">
            <div className="max-w-2xl">
              <p className="text-[11px] font-medium text-primary uppercase tracking-widest mb-3">ThermaScope AI</p>
              <h2 className="text-2xl font-semibold tracking-tight text-foreground leading-snug">
                Physics-Aware Thermal Satellite Intelligence
              </h2>
              <p className="text-sm text-muted mt-3 leading-relaxed max-w-lg">
                Landsat 8/9 Collection 2 Level-2 processing — ST_B10 thermal super-resolution
                and guided colorization via SwinIR architecture.
              </p>
              <div className="flex gap-3 mt-6">
                <Link to="/datasets">
                  <Button>Browse Datasets <ArrowRight className="h-4 w-4" /></Button>
                </Link>
                <Link to="/models">
                  <Button variant="secondary">View Pipeline</Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { label: 'Downloaded Scenes', value: stats.downloadedScenes, sub: 'Landsat C2 L2' },
            { label: 'Processed Scenes', value: stats.processedScenes, sub: formatDateTime(stats.lastProcessedAt) },
            { label: 'Training Status', value: stats.trainingStatus, sub: `Epoch 142/200` },
            { label: 'Current Model', value: stats.currentModel, sub: stats.modelVersion },
          ].map((stat) => (
            <Card key={stat.label} className="p-6">
              <p className="text-[11px] font-medium text-muted-dim uppercase tracking-wider">{stat.label}</p>
              <p className="text-2xl font-semibold mt-2 capitalize">{stat.value}</p>
              <p className="text-xs text-muted mt-1 truncate">{stat.sub}</p>
            </Card>
          ))}
        </div>
      </FadeIn>

      <div className="grid gap-6 lg:grid-cols-5">
        <FadeIn delay={0.1} className="lg:col-span-3">
          <Card>
            <div className="p-6 border-b border-border">
              <h3 className="text-sm font-semibold">Recent Processing Activity</h3>
            </div>
            <div className="divide-y divide-border">
              {activities.slice(0, 5).map((act) => (
                <div key={act.id} className="flex items-start gap-4 px-6 py-4 hover:bg-surface/50 transition-colors">
                  <div className={cn(
                    'mt-0.5 h-2 w-2 rounded-full shrink-0',
                    act.status === 'success' ? 'bg-success' : act.status === 'warning' ? 'bg-warning' : act.status === 'error' ? 'bg-error' : 'bg-primary'
                  )} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground leading-snug">{act.message}</p>
                    <p className="text-xs text-muted-dim mt-1">{formatDateTime(act.timestamp)}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </FadeIn>

        <FadeIn delay={0.15} className="lg:col-span-2 space-y-6">
          <Card>
            <div className="p-6 border-b border-border">
              <h3 className="text-sm font-semibold">System Health</h3>
            </div>
            <div className="p-4 space-y-2">
              {health.map((item) => (
                <div key={item.component} className="flex items-center justify-between rounded-lg px-3 py-2.5 hover:bg-surface transition-colors">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className={cn('h-1.5 w-1.5 rounded-full shrink-0',
                      item.status === 'healthy' ? 'bg-success' : item.status === 'degraded' ? 'bg-warning' : 'bg-error'
                    )} />
                    <div className="min-w-0">
                      <p className="text-xs font-medium truncate">{item.component}</p>
                      <p className="text-[10px] text-muted-dim truncate">{item.message}</p>
                    </div>
                  </div>
                  {item.latencyMs && <span className="text-[10px] font-mono text-muted-dim">{item.latencyMs}ms</span>}
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <div className="p-6 border-b border-border">
              <h3 className="text-sm font-semibold">Pipeline Status</h3>
            </div>
            <div className="p-6 space-y-4">
              {stats.pipelineStages.map((stage, i) => {
                const Icon = stageIcons[stage.status]
                return (
                  <div key={stage.id} className="flex items-center gap-3">
                    <div className={cn('flex h-7 w-7 items-center justify-center rounded-lg',
                      stage.status === 'completed' ? 'bg-success/10 text-success'
                      : stage.status === 'running' ? 'bg-warning/10 text-warning'
                      : 'bg-surface text-muted-dim'
                    )}>
                      <Icon className={cn('h-3.5 w-3.5', stage.status === 'running' && 'animate-spin')} />
                    </div>
                    <div className="flex-1">
                      <p className="text-xs font-medium">{stage.name}</p>
                    </div>
                    <Badge variant={stage.status === 'completed' ? 'success' : stage.status === 'running' ? 'warning' : 'outline'}>
                      {stage.status}
                    </Badge>
                    {i < stats.pipelineStages.length - 1 && (
                      <div className="absolute left-[46px] mt-8 h-4 w-px bg-border hidden" />
                    )}
                  </div>
                )
              })}
            </div>
            <div className="px-6 pb-6">
              <div className="flex justify-between text-xs text-muted mb-2">
                <span>Storage</span>
                <span>{formatBytes(stats.storageUsedBytes)}</span>
              </div>
              <div className="h-1 rounded-full bg-border overflow-hidden">
                <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${storagePct}%` }} />
              </div>
            </div>
          </Card>
        </FadeIn>
      </div>
    </div>
  )
}
