import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { api, artifactUrl, type FrozenResults, type RunManifest } from '@/lib/api'
import { FrozenDisclaimer } from '@/components/thermascope/FrozenDisclaimer'
import { Loader2 } from 'lucide-react'

export function ResultsPage() {
  const [params] = useSearchParams()
  const [experiment, setExperiment] = useState<'multiscene' | 'physics'>('physics')
  const [role, setRole] = useState<'validation' | 'test_locked'>('test_locked')
  const [frozen, setFrozen] = useState<FrozenResults | null>(null)
  const [runs, setRuns] = useState<RunManifest[]>([])
  const [runId, setRunId] = useState(params.get('run') ?? '')
  const [run, setRun] = useState<RunManifest | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.history().then((r) => {
      setRuns(r.filter((x) => x.output_root_key === 'ui_runs' || x.run_id?.includes('T')))
      if (!runId) {
        const ui = r.find((x) => x.output_root_key === 'ui_runs' && x.status === 'completed')
        if (ui?.run_id) setRunId(ui.run_id)
      }
    }).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    setLoading(true)
    api.frozenResults(experiment, role).then(setFrozen).finally(() => setLoading(false))
  }, [experiment, role])

  useEffect(() => {
    if (!runId) return
    api.run(runId).then(setRun).catch(() => setRun(null))
  }, [runId])

  if (loading) return <Loader2 className="mx-auto h-8 w-8 animate-spin text-accent" />

  const chartData =
    frozen?.per_scene.map((s) => ({
      scene: s.scene_id.split('_').slice(-2)[0] ?? s.scene_id,
      bicubic: s.metrics.bicubic?.rmse_kelvin,
      swinir: s.metrics.swinir_pretrained?.rmse_kelvin,
      refined: s.metrics.swinir_consistency_refined?.rmse_kelvin,
    })) ?? []

  const comparison = run?.sr_comparisons?.[0]
  const rootKey = run?.output_root_key ?? 'ui_runs'

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-text">Results Lab</h1>
        <p className="mt-1 text-sm text-muted">Frozen benchmark evidence and current-run inference metrics.</p>
      </div>

      <FrozenDisclaimer />

      <section className="rounded-xl border border-border-slate bg-surface p-6 shadow-glow space-y-4">
        <div className="flex flex-wrap gap-3 items-center">
          <h2 className="text-sm font-semibold text-text flex-1">A — Frozen experimental results</h2>
          {frozen?.locked && (
            <span className="rounded-md bg-warning/10 px-2 py-1 text-[10px] font-semibold uppercase text-warning border border-warning/30">
              Locked test — not for tuning
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            className="rounded-lg border border-border-slate bg-midnight px-3 py-1.5 text-xs"
            value={experiment}
            onChange={(e) => setExperiment(e.target.value as 'multiscene' | 'physics')}
          >
            <option value="multiscene">Multiscene baseline</option>
            <option value="physics">Physics refinement</option>
          </select>
          <select
            className="rounded-lg border border-border-slate bg-midnight px-3 py-1.5 text-xs"
            value={role}
            onChange={(e) => setRole(e.target.value as 'validation' | 'test_locked')}
          >
            <option value="validation">Validation (Kerala)</option>
            <option value="test_locked">Locked test (Deccan + Assam)</option>
          </select>
        </div>
        <p className="text-xs text-muted">{frozen?.disclaimer}</p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <XAxis dataKey="scene" tick={{ fill: '#8b9cb3', fontSize: 10 }} />
              <YAxis tick={{ fill: '#8b9cb3', fontSize: 10 }} label={{ value: 'RMSE (K)', angle: -90, position: 'insideLeft', fill: '#8b9cb3' }} />
              <Tooltip contentStyle={{ background: '#141c2b', border: '1px solid #243044' }} />
              <Legend />
              <Bar dataKey="bicubic" fill="#64748b" name="Bicubic" />
              <Bar dataKey="swinir" fill="#22d3ee" name="SwinIR" />
              {chartData.some((d) => d.refined != null) && <Bar dataKey="refined" fill="#f97316" name="Refined" />}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted-dim border-b border-border-slate">
                <th className="py-2 pr-4">Scene</th>
                {frozen?.methods.map((m) => (
                  <th key={m} className="py-2 pr-4">
                    {m} RMSE (K)
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {frozen?.per_scene.map((s) => (
                <tr key={s.scene_id} className="border-b border-border-slate/50">
                  <td className="py-2 pr-4 font-mono text-[10px]">{s.scene_id}</td>
                  {frozen.methods.map((m) => (
                    <td key={m} className="py-2 pr-4 font-mono">
                      {s.metrics[m]?.rmse_kelvin?.toFixed(4) ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-xl border border-border-slate bg-surface p-6 shadow-glow space-y-4">
        <h2 className="text-sm font-semibold text-text">B — Current run results</h2>
        <select
          className="rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs max-w-md"
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
        >
          <option value="">Select UI run…</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} — {r.status}
            </option>
          ))}
        </select>

        {!run && <p className="text-sm text-muted">No run selected.</p>}

        {run && !comparison && (
          <p className="text-sm text-amber-200/90">Metrics unavailable — no evaluation reference supplied for this run.</p>
        )}

        {comparison && (
          <>
            <p className="text-xs text-muted">{comparison.metrics_note}</p>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-muted-dim border-b border-border-slate">
                  <th className="py-2 text-left">Method</th>
                  <th className="py-2">RMSE (K)</th>
                  <th className="py-2">MAE (K)</th>
                  <th className="py-2">PSNR</th>
                  <th className="py-2">SSIM</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.metrics_vs_hr_reference).map(([method, m]) =>
                  m ? (
                    <tr key={method} className="border-b border-border-slate/40">
                      <td className="py-2">{method}</td>
                      <td className="py-2 font-mono text-center">{m.rmse_kelvin?.toFixed(4)}</td>
                      <td className="py-2 font-mono text-center">{m.mae_kelvin?.toFixed(4)}</td>
                      <td className="py-2 font-mono text-center">{Number.isFinite(m.psnr!) ? m.psnr!.toFixed(2) : '∞'}</td>
                      <td className="py-2 font-mono text-center">{m.ssim?.toFixed(4)}</td>
                    </tr>
                  ) : null
                )}
              </tbody>
            </table>
          </>
        )}

        {run?.patches?.[0]?.colorization_outputs?.A_standard_colormap && run.run_id && (
          <img
            className="max-w-sm rounded-lg border border-border-slate"
            src={artifactUrl(
              rootKey,
              rootKey === 'ui_runs'
                ? `${run.run_id}/${run.patches[0].colorization_outputs.A_standard_colormap.replace(/\\/g, '/')}`
                : run.patches[0].colorization_outputs.A_standard_colormap.replace(/\\/g, '/')
            )}
            alt="Run output"
          />
        )}
      </section>
    </div>
  )
}
