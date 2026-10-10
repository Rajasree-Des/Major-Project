import { Loader2 } from 'lucide-react'
import type { RunManifest } from '@/lib/api'

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  validating: 'Validating input',
  loading_model: 'Loading model',
  preparing_input: 'Preparing thermal input',
  swinir_inference: 'Running SwinIR inference',
  consistency_refinement: 'Applying consistency refinement',
  colorization: 'Generating visualizations',
  computing_metrics: 'Computing metrics & stage previews',
  saving_artifacts: 'Saving artifacts',
  completed: 'Completed',
  failed: 'Failed',
}

type Props = {
  job: RunManifest | null
  elapsedSec: number
}

export function JobProgressPanel({ job, elapsedSec }: Props) {
  if (!job) return null
  const stage = job.stage ?? job.status
  const label = STAGE_LABELS[stage] ?? job.progress_message ?? stage
  const running = job.status === 'running' || job.status === 'queued'

  return (
    <div className="rounded-xl border border-border-slate bg-surface p-4 shadow-glow">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-dim">Pipeline status</p>
          <p className="text-sm font-medium text-text mt-0.5">{label}</p>
        </div>
        {running && <Loader2 className="h-5 w-5 animate-spin text-accent shrink-0" />}
      </div>
      <p className="mt-2 text-xs text-muted font-mono">Elapsed {elapsedSec.toFixed(0)}s</p>
      {job.error && <p className="mt-2 text-sm text-error">{job.error}</p>}
      {job.logs && job.logs.length > 0 && (
        <pre className="mt-3 max-h-32 overflow-auto rounded-lg bg-midnight p-3 text-[10px] font-mono text-muted leading-relaxed">
          {job.logs.slice(-12).join('\n')}
        </pre>
      )}
    </div>
  )
}
