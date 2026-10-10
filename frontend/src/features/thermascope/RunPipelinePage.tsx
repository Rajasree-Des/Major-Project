import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type SceneSummary } from '@/lib/api'
import { PageHeader } from '@/components/shared/PageHeader'
import type { ReactNode } from 'react'
import { Loader2, Lock } from 'lucide-react'

export function RunPipelinePage() {
  const [scenes, setScenes] = useState<SceneSummary[]>([])
  const [mode, setMode] = useState<'single' | 'validation_sample'>('single')
  const [sceneId, setSceneId] = useState('LC08_L2SP_144053_20250225_02_T1')
  const [patchId, setPatchId] = useState('000096')
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Record<string, unknown> | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [frozen, setFrozen] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    api.scenes().then(setScenes)
    api.overview().then((o) => setFrozen(o.frozen_config_summary))
  }, [])

  useEffect(() => {
    if (!jobId) return
    const t = setInterval(async () => {
      try {
        const j = await api.job(jobId)
        setJob(j as Record<string, unknown>)
        if (j.status === 'completed' || j.status === 'failed') clearInterval(t)
      } catch {
        /* keep polling */
      }
    }, 1500)
    return () => clearInterval(t)
  }, [jobId])

  async function start() {
    setSubmitting(true)
    setError(null)
    setJob(null)
    try {
      const res = await api.submitJob({
        mode,
        scene_id: sceneId,
        patch_id: mode === 'single' ? patchId : undefined,
        num_patches: 4,
        seed: 42,
      })
      setJobId(res.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const status = job?.status as string | undefined
  const progress = (job?.progress_pct as number) ?? 0

  return (
    <div>
      <PageHeader
        title="Run Pipeline"
        description="CPU inference: synthetic LR → SwinIR-S → consistency refinement (β=1.0) → Kelvin → colorization. No training."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-workspace-border bg-white p-6 shadow-panel space-y-5">
          <div>
            <label className="text-xs font-semibold uppercase text-workspace-muted-fg">Workflow</label>
            <div className="mt-2 flex gap-2">
              <ModeBtn active={mode === 'single'} onClick={() => setMode('single')}>
                Single patch
              </ModeBtn>
              <ModeBtn active={mode === 'validation_sample'} onClick={() => setMode('validation_sample')}>
                Validation sample (4 patches)
              </ModeBtn>
            </div>
          </div>

          {mode === 'single' && (
            <>
              <Field label="Scene">
                <select className="w-full rounded-xl border px-3 py-2 text-sm" value={sceneId} onChange={(e) => setSceneId(e.target.value)}>
                  {scenes.map((s) => (
                    <option key={s.scene_id} value={s.scene_id}>
                      {s.scene_id}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Patch ID">
                <input
                  className="w-full rounded-xl border px-3 py-2 font-mono text-sm"
                  value={patchId}
                  onChange={(e) => setPatchId(e.target.value)}
                />
              </Field>
            </>
          )}

          <button
            type="button"
            disabled={submitting || status === 'running' || status === 'queued'}
            onClick={start}
            className="w-full rounded-xl bg-accent py-3 text-sm font-semibold text-white hover:bg-accent-dim disabled:opacity-50"
          >
            {submitting ? 'Submitting…' : 'Start inference'}
          </button>
          {error && <p className="text-sm text-error">{error}</p>}
        </div>

        <div className="rounded-2xl border border-workspace-border bg-white p-6 shadow-panel">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Lock className="h-4 w-4 text-accent" /> Frozen configuration
          </h3>
          <pre className="mt-3 text-xs bg-workspace-muted rounded-xl p-4 overflow-auto max-h-48 font-mono text-workspace-muted-fg">
            {JSON.stringify(frozen, null, 2)}
          </pre>
          <p className="mt-3 text-xs text-workspace-muted-fg leading-relaxed">
            Degradation: scale 2, blur σ 1.2 HR px, 0.05 K noise, seed 42. Colormap 295–320 K inferno. Parameters are not editable in this demo UI.
          </p>
        </div>
      </div>

      {jobId && (
        <section className="mt-8 rounded-2xl border border-workspace-border bg-white p-6 shadow-panel">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold">Job {jobId}</p>
              <p className="text-xs text-workspace-muted-fg capitalize mt-0.5">Status: {status ?? '…'}</p>
            </div>
            {status === 'running' && <Loader2 className="h-5 w-5 animate-spin text-accent" />}
          </div>
          <div className="mt-4 h-2 rounded-full bg-workspace-border overflow-hidden">
            <div className="h-full bg-accent transition-all duration-500" style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-2 text-xs text-workspace-muted-fg">{String(job?.progress_message ?? '')}</p>
          {status === 'failed' && <p className="mt-2 text-sm text-error">{String(job?.error)}</p>}
          {status === 'completed' && (
            <div className="mt-4 flex gap-4 text-sm">
              <Link to={`/results?run=${jobId}`} className="text-accent font-medium hover:underline">
                View results →
              </Link>
              <Link to={`/colorization?run=${jobId}`} className="text-accent font-medium hover:underline">
                Colorization studio →
              </Link>
            </div>
          )}
          {job && (job.logs as string[] | undefined)?.length ? (
            <pre className="mt-4 max-h-40 overflow-auto rounded-xl bg-slate-900 text-slate-100 p-4 text-xs font-mono">
              {(job.logs as string[]).join('\n')}
            </pre>
          ) : null}
        </section>
      )}
    </div>
  )
}

function ModeBtn({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-xl px-3 py-2 text-xs font-medium border ${
        active ? 'border-accent bg-accent/10 text-accent-dim' : 'border-workspace-border text-workspace-muted-fg'
      }`}
    >
      {children}
    </button>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label className="text-xs font-semibold uppercase text-workspace-muted-fg">{label}</label>
      <div className="mt-1.5">{children}</div>
    </div>
  )
}
