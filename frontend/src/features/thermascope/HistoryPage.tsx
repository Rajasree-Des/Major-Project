import { Link } from 'react-router-dom'
import { api, type RunManifest } from '@/lib/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Loader2 } from 'lucide-react'

export function HistoryPage() {
  const { data: runs, loading, error } = useAsyncData(() => api.history(), [])

  if (loading) return <Loader2 className="mx-auto h-8 w-8 animate-spin text-accent" />
  if (error) return <p className="text-error text-sm">{error}</p>

  const list = runs ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text">Experiment History</h1>
        <p className="mt-1 text-sm text-muted">Manifest-backed runs only.</p>
      </div>

      {!list.length && (
        <div className="rounded-xl border border-dashed border-border-slate p-12 text-center">
          <p className="text-muted text-sm">No runs yet.</p>
          <Link to="/studio" className="mt-4 inline-block text-accent text-sm font-medium hover:underline">
            Start first experiment →
          </Link>
        </div>
      )}

      <div className="space-y-3">
        {list.map((r: RunManifest) => (
          <article key={r.run_id} className="rounded-xl border border-border-slate bg-surface p-5 shadow-glow">
            <div className="flex flex-wrap justify-between gap-3">
              <div>
                <p className="font-mono text-sm text-text">{r.run_id}</p>
                <p className="text-xs text-muted capitalize mt-1">{r.status} · {r.stage ?? ''}</p>
              </div>
              <div className="flex gap-3 text-xs">
                <Link to={`/studio`} className="text-accent hover:underline">Studio</Link>
                <Link to={`/results?run=${encodeURIComponent(r.run_id!)}`} className="text-accent hover:underline">Results</Link>
                <Link to={`/colorization?run=${encodeURIComponent(r.run_id!)}`} className="text-accent hover:underline">Colorization</Link>
              </div>
            </div>
            <dl className="mt-3 grid grid-cols-3 gap-2 text-xs text-muted">
              <div><dt className="text-muted-dim">Runtime</dt><dd>{r.runtime_s != null ? `${Number(r.runtime_s).toFixed(1)}s` : '—'}</dd></div>
              <div><dt className="text-muted-dim">Patches</dt><dd>{r.patches?.length ?? r.patch_ids?.length ?? '—'}</dd></div>
              <div><dt className="text-muted-dim">β</dt><dd>{(r as { refinement_beta?: number }).refinement_beta ?? '—'}</dd></div>
            </dl>
          </article>
        ))}
      </div>
    </div>
  )
}
