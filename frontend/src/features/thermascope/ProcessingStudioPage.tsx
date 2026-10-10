import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { api, artifactUrl, previewUrl, type RunManifest, type SceneSummary } from '@/lib/api'
import { useSelection } from '@/context/SelectionContext'
import { FrozenDisclaimer } from '@/components/thermascope/FrozenDisclaimer'
import { ImageViewer } from '@/components/thermascope/ImageViewer'
import { CompareSlider } from '@/components/thermascope/CompareSlider'
import { JobProgressPanel } from '@/components/thermascope/JobProgressPanel'
import { Play, Lock } from 'lucide-react'

const TABS = [
  { id: 'preview_thermal', label: 'Reference thermal' },
  { id: 'preview_rgb', label: 'Reference RGB' },
  { id: 'input_lr', label: 'LR input' },
  { id: 'bicubic', label: 'Bicubic' },
  { id: 'swinir', label: 'SwinIR' },
  { id: 'refined', label: 'Refined' },
  { id: 'color_a', label: 'Colormap A' },
  { id: 'compare', label: 'Compare' },
] as const

export function ProcessingStudioPage() {
  const [params] = useSearchParams()
  const { sceneId, patchId, setSelection } = useSelection()
  const [scenes, setScenes] = useState<SceneSummary[]>([])
  const [patches, setPatches] = useState<string[]>([])
  const [patchSearch, setPatchSearch] = useState('')
  const [frozen, setFrozen] = useState<Record<string, unknown> | null>(null)
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('preview_thermal')
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<RunManifest | null>(null)
  const [activeRun, setActiveRun] = useState<RunManifest | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [t0, setT0] = useState<number | null>(null)

  useEffect(() => {
    const s = params.get('scene')
    const p = params.get('patch')
    if (s && p) setSelection(s, p)
  }, [params, setSelection])

  useEffect(() => {
    api.scenes().then(setScenes)
    api.overview().then((o) => setFrozen(o.frozen_config_summary))
  }, [])

  useEffect(() => {
    if (!sceneId) return
    api.patches(sceneId, 600, 0).then((r) => setPatches(r.patch_ids))
  }, [sceneId])

  useEffect(() => {
    if (!jobId) return
    setT0(Date.now())
    const poll = setInterval(async () => {
      try {
        const j = await api.job(jobId)
        setJob(j)
        if (j.status === 'completed' || j.status === 'failed') {
          clearInterval(poll)
          if (j.status === 'completed') setActiveRun(j)
        }
      } catch {
        /* retry */
      }
    }, 1200)
    return () => clearInterval(poll)
  }, [jobId])

  useEffect(() => {
    if (!t0 || !job || (job.status !== 'running' && job.status !== 'queued')) return
    const t = setInterval(() => setElapsed((Date.now() - t0) / 1000), 500)
    return () => clearInterval(t)
  }, [t0, job])

  async function runPipeline() {
    setSubmitting(true)
    setError(null)
    setJob(null)
    setActiveRun(null)
    setElapsed(0)
    try {
      const res = await api.submitJob({ mode: 'single', scene_id: sceneId, patch_id: patchId })
      setJobId(res.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const run = activeRun
  const rootKey = run?.output_root_key ?? 'ui_runs'
  const rid = run?.run_id ?? jobId ?? ''
  const stages = run?.stage_artifacts ?? {}
  const colorOut = run?.patches?.[0]?.colorization_outputs

  function stageUrl(key: string) {
    const rel = stages[key]
    if (!rel || !rid) return null
    return artifactUrl(rootKey, `${rid}/${rel}`)
  }

  function tabImageSrc(): string | null {
    switch (tab) {
      case 'preview_thermal':
        return previewUrl(sceneId, patchId, 'thermal_colormap')
      case 'preview_rgb':
        return previewUrl(sceneId, patchId, 'rgb')
      case 'input_lr':
        return stageUrl('input_lr')
      case 'bicubic':
        return stageUrl('bicubic')
      case 'swinir':
        return stageUrl('swinir')
      case 'refined':
        return stageUrl('refined')
      case 'color_a':
        if (colorOut?.A_standard_colormap && rid)
          return artifactUrl(rootKey, rootKey === 'ui_runs' ? `${rid}/${colorOut.A_standard_colormap.replace(/\\/g, '/')}` : colorOut.A_standard_colormap.replace(/\\/g, '/'))
        return stageUrl('refined')
      default:
        return null
    }
  }

  const filtered = patches.filter((p) => p.includes(patchSearch.replace(/\D/g, '').padStart(6, '0').slice(-6) || patchSearch))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-text">Processing Studio</h1>
        <p className="mt-1 text-sm text-muted max-w-2xl">
          Live CPU pipeline: synthetic LR → SwinIR-S ×2 → consistency refinement (β=1.0) → colorization.
        </p>
      </div>

      <FrozenDisclaimer />

      <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
        <div className="space-y-4">
          <div className="rounded-xl border border-border-slate bg-surface p-4 shadow-glow space-y-3">
            <label className="text-[10px] font-semibold uppercase tracking-wide text-muted-dim">Scene</label>
            <select
              className="w-full rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs font-mono text-text"
              value={sceneId}
              onChange={(e) => setSelection(e.target.value, patchId)}
            >
              {scenes.map((s) => (
                <option key={s.scene_id} value={s.scene_id}>
                  {s.split_role} · {s.region?.slice(0, 36) ?? s.scene_id}
                </option>
              ))}
            </select>
            <label className="text-[10px] font-semibold uppercase tracking-wide text-muted-dim">Patch ID</label>
            <input
              className="w-full rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs font-mono"
              placeholder="Search patch…"
              value={patchSearch || patchId}
              onChange={(e) => setPatchSearch(e.target.value)}
              onBlur={() => {
                const d = patchSearch.replace(/\D/g, '')
                if (d) setSelection(sceneId, d.padStart(6, '0'))
              }}
            />
            <select
              className="w-full rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs font-mono max-h-24"
              size={4}
              value={patchId}
              onChange={(e) => setSelection(sceneId, e.target.value)}
            >
              {(patchSearch ? filtered : patches.slice(0, 120)).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={submitting || job?.status === 'running'}
              onClick={runPipeline}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent py-3 text-sm font-semibold text-midnight hover:bg-accent-dim disabled:opacity-50 transition-colors"
            >
              <Play className="h-4 w-4" /> Run pipeline
            </button>
            {error && <p className="text-xs text-error">{error}</p>}
            {job && (job.status === 'running' || job.status === 'queued' || job.status === 'failed') && (
              <JobProgressPanel job={job} elapsedSec={elapsed} />
            )}
            {run?.status === 'completed' && (
              <div className="text-xs text-success">
                Done in {Number(run.runtime_s).toFixed(1)}s ·{' '}
                <Link to={`/results?run=${run.run_id}`} className="text-accent underline">
                  Results
                </Link>
              </div>
            )}
          </div>

          <div className="rounded-xl border border-border-slate bg-surface p-4">
            <h3 className="flex items-center gap-2 text-xs font-semibold text-text">
              <Lock className="h-3.5 w-3.5 text-accent" /> Frozen configuration
            </h3>
            <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-midnight p-2 text-[10px] font-mono text-muted">
              {JSON.stringify(frozen, null, 2)}
            </pre>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex flex-wrap gap-1 border-b border-border-slate pb-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-colors ${
                  tab === t.id ? 'bg-accent/15 text-accent' : 'text-muted hover:text-text'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
              {tab === 'compare' ? (
                stageUrl('bicubic') && stageUrl('refined') ? (
                  <CompareSlider beforeSrc={stageUrl('bicubic')!} afterSrc={stageUrl('refined')!} beforeLabel="Bicubic" afterLabel="Refined" />
                ) : (
                  <p className="text-sm text-muted py-12 text-center">Run pipeline to enable bicubic vs refined comparison.</p>
                )
              ) : tabImageSrc() ? (
                <ImageViewer src={tabImageSrc()!} alt={tab} downloadName={`thermascope_${tab}.png`} />
              ) : (
                <p className="text-sm text-muted py-12 text-center">
                  {tab.startsWith('preview_') ? 'Loading reference preview…' : 'Run pipeline to generate this stage output.'}
                </p>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
