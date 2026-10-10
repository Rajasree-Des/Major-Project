import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { api, artifactUrl, previewUrl, type Overview } from '@/lib/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { DemoGuide } from '@/components/thermascope/DemoGuide'
import { FrozenDisclaimer } from '@/components/thermascope/FrozenDisclaimer'
import { AlertCircle, ArrowRight, Play, BarChart3, Loader2 } from 'lucide-react'

const WORKFLOW = [
  'Landsat QA input',
  'Synthetic LR',
  'SwinIR ×2',
  'Consistency refine',
  'RGB-guided viz',
]

export function OverviewPage() {
  const { data, loading, error } = useAsyncData(() => api.overview(), [])

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-10 w-10 animate-spin text-accent" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-error/40 bg-error/10 p-6 flex gap-3 text-error">
        <AlertCircle className="shrink-0" />
        <div>
          <p className="font-medium">Backend unavailable</p>
          <p className="text-sm mt-1 opacity-90">{error ?? 'Run: py -m backend'}</p>
        </div>
      </div>
    )
  }

  const o = data as Overview
  const studioHref =
    o.demo_patch_available && o.demo_scene_id && o.demo_patch_id
      ? `/studio?scene=${encodeURIComponent(o.demo_scene_id)}&patch=${encodeURIComponent(o.demo_patch_id)}`
      : '/studio'

  const heroThermal = o.demo_patch_available
    ? previewUrl(o.demo_scene_id!, o.demo_patch_id!, 'thermal_colormap')
    : null
  const heroIntegrated = artifactUrl(
    'integrated_pipeline_v1',
    'patches/LC08_L2SP_144053_20250225_02_T1/patch_000096/A_standard_colormap.png'
  )

  return (
    <div className="space-y-10">
      <section className="relative overflow-hidden rounded-2xl border border-border-slate bg-surface shadow-glow">
        <div className="grid lg:grid-cols-2 gap-8 p-8 lg:p-10">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-accent font-medium">ThermaScope</p>
            <h1 className="mt-3 text-3xl lg:text-4xl font-bold tracking-tight text-text leading-tight">
              See thermal Earth data in greater detail.
            </h1>
            <p className="mt-4 text-sm text-muted leading-relaxed max-w-lg">{o.subtitle ?? o.tagline}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                to={studioHref}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-midnight hover:bg-accent-dim transition-colors"
              >
                <Play className="h-4 w-4" /> Launch Processing Studio
              </Link>
              <Link
                to="/results"
                className="inline-flex items-center gap-2 rounded-lg border border-border-slate px-5 py-2.5 text-sm font-medium text-text hover:bg-surface-elevated"
              >
                <BarChart3 className="h-4 w-4" /> Explore Results
              </Link>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 min-h-[200px]">
            {heroThermal && (
              <motion.img
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                src={heroThermal}
                alt="Reference thermal preview"
                className="rounded-lg border border-border-slate object-cover h-full w-full max-h-[220px]"
              />
            )}
            <motion.img
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.05 }}
              src={heroIntegrated}
              alt="Refined colormap output"
              className="rounded-lg border border-border-slate object-cover h-full w-full max-h-[220px]"
              onError={(e) => {
                ;(e.target as HTMLImageElement).style.display = 'none'
              }}
            />
          </div>
        </div>
      </section>

      <FrozenDisclaimer />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <Stat label="Prepared scenes" value={String(o.scenes_prepared)} />
        <Stat label="Total patches" value={String(o.total_patches ?? '—')} />
        <Stat label="Device" value="CPU" />
        <Stat label="Refinement β" value={String(o.frozen_config_summary.refinement_beta)} />
        <Stat label="Checkpoint" value={o.checkpoint.available ? 'Ready' : 'Missing'} warn={!o.checkpoint.available} />
      </div>

      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-dim mb-4">Research workflow</h2>
        <div className="flex flex-wrap items-center gap-2">
          {WORKFLOW.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <span className="rounded-md bg-surface-elevated border border-border-slate px-3 py-1.5 text-xs text-text">{step}</span>
              {i < WORKFLOW.length - 1 && <ArrowRight className="h-3 w-3 text-muted-dim" />}
            </div>
          ))}
        </div>
      </section>

      <DemoGuide demoScene={o.demo_scene_id} demoPatch={o.demo_patch_id} demoAvailable={o.demo_patch_available} />

      {o.latest_run && (
        <section className="rounded-xl border border-border-slate bg-surface p-4 text-xs font-mono text-muted overflow-x-auto">
          <p className="text-[10px] uppercase text-muted-dim mb-2">Most recent run (manifest summary)</p>
          <pre>{JSON.stringify({ run_id: (o.latest_run as { run_id?: string }).run_id, status: (o.latest_run as { status?: string }).status }, null, 2)}</pre>
        </section>
      )}
    </div>
  )
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="rounded-xl border border-border-slate bg-surface p-4 shadow-glow">
      <p className="text-[10px] uppercase tracking-wide text-muted-dim">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${warn ? 'text-warning' : 'text-text'}`}>{value}</p>
    </div>
  )
}
