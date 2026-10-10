import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, artifactUrl, previewUrl, type RunManifest } from '@/lib/api'
import { ImageViewer } from '@/components/thermascope/ImageViewer'
import { FrozenDisclaimer } from '@/components/thermascope/FrozenDisclaimer'
import { Loader2 } from 'lucide-react'

const METHODS = [
  { key: 'A_standard_colormap', legend: 'A_standard_colormap_legend', label: 'A — Standard colormap' },
  { key: 'B_rgb_guided', legend: 'B_rgb_guided_legend', label: 'B — RGB-guided' },
  { key: 'C_rgb_overlay', legend: 'C_rgb_overlay_legend', label: 'C — RGB overlay' },
] as const

export function ColorizationStudioPage() {
  const [params] = useSearchParams()
  const [runs, setRuns] = useState<RunManifest[]>([])
  const [runId, setRunId] = useState(params.get('run') ?? '')
  const [run, setRun] = useState<RunManifest | null>(null)
  const [method, setMethod] = useState(0)
  const [patchIdx, setPatchIdx] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.history().then((r) => {
      const done = r.filter((x) => x.status === 'completed')
      setRuns(done)
      if (params.get('run')) setRunId(params.get('run')!)
      else if (!runId && done[0]?.run_id) setRunId(done[0].run_id)
    }).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!runId) return
    api.run(runId).then(setRun)
  }, [runId])

  if (loading) return <Loader2 className="mx-auto h-8 w-8 animate-spin text-accent" />

  const rootKey = run?.output_root_key ?? 'ui_runs'
  const patch = run?.patches?.[patchIdx]
  const m = METHODS[method]
  const rel = patch?.colorization_outputs?.[m.key]?.replace(/\\/g, '/')
  const imgPath = rel && run?.run_id ? (rootKey === 'ui_runs' ? `${run.run_id}/${rel}` : rel) : null
  const legendRel = rel?.replace('.png', '_legend.png')
  const legendPath = legendRel && run?.run_id ? (rootKey === 'ui_runs' ? `${run.run_id}/${legendRel}` : legendRel) : null
  const kelvinRel = patch?.thermal_kelvin_file?.replace(/\\/g, '/')
  const kelvinPath = kelvinRel && run?.run_id && rootKey === 'ui_runs' ? `${run.run_id}/${kelvinRel}` : kelvinRel

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text">Colorization Studio</h1>
        <p className="mt-1 text-sm text-muted">Visualization-only RGB guidance; Kelvin arrays stored separately.</p>
      </div>
      <FrozenDisclaimer />

      <div className="flex flex-wrap gap-3">
        <select className="rounded-lg border border-border-slate bg-surface px-3 py-2 text-xs" value={runId} onChange={(e) => setRunId(e.target.value)}>
          <option value="">Select run…</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
          ))}
        </select>
        {run && (run.patches?.length ?? 0) > 1 && (
          <select className="rounded-lg border border-border-slate bg-surface px-3 py-2 text-xs" value={patchIdx} onChange={(e) => setPatchIdx(Number(e.target.value))}>
            {run.patches!.map((p, i) => (
              <option key={p.patch_id} value={i}>{p.patch_id}</option>
            ))}
          </select>
        )}
        <div className="flex gap-1">
          {METHODS.map((x, i) => (
            <button
              key={x.key}
              type="button"
              onClick={() => setMethod(i)}
              className={`rounded-lg px-3 py-2 text-[11px] font-medium ${i === method ? 'bg-accent/15 text-accent' : 'text-muted border border-border-slate'}`}
            >
              {x.label.split('—')[0]}
            </button>
          ))}
        </div>
      </div>

      {!imgPath ? (
        <p className="text-sm text-muted">No colorization outputs — run the pipeline first.</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <ImageViewer src={artifactUrl(rootKey, imgPath)} alt={m.label} downloadName={`${m.key}.png`} />
          {legendPath && <ImageViewer src={artifactUrl(rootKey, legendPath)} alt="Legend" />}
          {patch && (
            <ImageViewer
              src={previewUrl(patch.scene_id, patch.patch_id, 'rgb')}
              alt="RGB context"
            />
          )}
        </div>
      )}

      {kelvinPath && (
        <p className="text-xs text-muted">
          Kelvin array:{' '}
          <a href={artifactUrl(rootKey, kelvinPath)} className="text-accent underline" download>
            download .npy
          </a>{' '}
          (not embedded in PNG)
        </p>
      )}
    </div>
  )
}
