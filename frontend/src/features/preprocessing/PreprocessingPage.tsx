import { useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown, ChevronUp, Terminal, CheckCircle2, Loader2, Clock } from 'lucide-react'
import { fetchPreprocessingSteps, fetchPreprocessingLogs } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { FadeIn } from '@/components/shared/PageTransition'
import { cn } from '@/lib/utils'
import type { ProcessingStepStatus } from '@/types'

const statusConfig: Record<ProcessingStepStatus, { icon: typeof CheckCircle2; color: string; bg: string }> = {
  completed: { icon: CheckCircle2, color: 'text-success', bg: 'bg-success/10 border-success/20' },
  running: { icon: Loader2, color: 'text-warning', bg: 'bg-warning/10 border-warning/20' },
  waiting: { icon: Clock, color: 'text-muted-dim', bg: 'bg-surface border-border' },
  pending: { icon: Clock, color: 'text-muted-dim', bg: 'bg-surface border-border' },
  failed: { icon: Clock, color: 'text-error', bg: 'bg-error/10 border-error/20' },
}

export function PreprocessingPage() {
  const [logsOpen, setLogsOpen] = useState(false)
  const { data: steps, loading } = useAsyncData(() => fetchPreprocessingSteps())
  const { data: logs } = useAsyncData(() => fetchPreprocessingLogs())

  if (loading || !steps) {
    return <div className="h-64 rounded-2xl border border-border animate-pulse" />
  }

  return (
    <div className="space-y-8">
      <FadeIn>
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Preprocessing Pipeline</h2>
            <p className="text-sm text-muted mt-1">Landsat Collection 2 Level-2 radiometric and geometric preparation</p>
          </div>
          <Button size="sm">Run Pipeline</Button>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="overflow-x-auto pb-4">
          <div className="flex items-stretch gap-3 min-w-max">
            {steps.map((step, i) => {
              const config = statusConfig[step.status]
              const Icon = config.icon
              return (
                <div key={step.id} className="flex items-center gap-3">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.06 }}
                    className={cn('w-52 rounded-xl border p-5 transition-all hover:shadow-elevated', config.bg,
                      step.status === 'running' && 'ring-1 ring-warning/30'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <Icon className={cn('h-4 w-4', config.color, step.status === 'running' && 'animate-spin')} />
                      <span className={cn('text-[10px] font-medium uppercase tracking-wider', config.color)}>
                        {step.status}
                      </span>
                    </div>
                    <h3 className="text-sm font-medium mb-1">{step.name}</h3>
                    <p className="text-[11px] text-muted leading-relaxed line-clamp-2">{step.description}</p>
                    {step.status === 'running' && (
                      <div className="mt-3">
                        <Progress value={step.progress} size="sm" />
                        <p className="text-[10px] text-muted-dim mt-1">{step.progress}%</p>
                      </div>
                    )}
                    {step.duration && step.status === 'completed' && (
                      <p className="text-[10px] text-muted-dim mt-2">{step.duration}</p>
                    )}
                  </motion.div>
                  {i < steps.length - 1 && (
                    <svg width="24" height="24" viewBox="0 0 24 24" className="text-muted-dim shrink-0">
                      <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <button
            onClick={() => setLogsOpen(!logsOpen)}
            className="flex w-full items-center justify-between px-6 py-4 hover:bg-surface transition-colors"
          >
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-muted" />
              <span className="text-sm font-medium">Processing Logs</span>
            </div>
            {logsOpen ? <ChevronUp className="h-4 w-4 text-muted" /> : <ChevronDown className="h-4 w-4 text-muted" />}
          </button>
          {logsOpen && logs && (
            <div className="border-t border-border bg-background px-6 py-4 font-mono text-[11px] text-muted leading-relaxed space-y-1 max-h-48 overflow-y-auto">
              {logs.map((line, i) => (
                <p key={i} className={line.includes('WARN') ? 'text-warning' : line.includes('ERROR') ? 'text-error' : ''}>{line}</p>
              ))}
            </div>
          )}
        </div>
      </FadeIn>
    </div>
  )
}
