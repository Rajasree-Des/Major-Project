import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, previewUrl, type SceneSummary } from '@/lib/api'
import { useSelection } from '@/context/SelectionContext'
import { ImageViewer } from '@/components/thermascope/ImageViewer'
import { AlertTriangle, Loader2 } from 'lucide-react'

export function SceneExplorerPage() {
  const { sceneId, patchId, setSelection } = useSelection()
  const [scenes, setScenes] = useState<SceneSummary[]>([])
  const [patchSearch, setPatchSearch] = useState('')
  const [patchOptions, setPatchOptions] = useState<string[]>([])
  const [meta, setMeta] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [sceneFilter, setSceneFilter] = useState('')

  useEffect(() => {
    api
      .scenes()
      .then((s) => {
        setScenes(s)
        if (!sceneId && s[0]) setSelection(s[0].scene_id, patchId)
      })
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!sceneId) return
    api.patches(sceneId, 800, 0).then((r) => setPatchOptions(r.patch_ids))
  }, [sceneId])

  useEffect(() => {
    if (!sceneId || !patchId) return
    api.patchMeta(sceneId, patchId).then(setMeta).catch((e) => setErr(String(e)))
  }, [sceneId, patchId])

  const filteredScenes = scenes.filter((s) => s.scene_id.toLowerCase().includes(sceneFilter.toLowerCase()))
  const filtered = patchOptions.filter((p) => p.includes(patchSearch.replace(/\D/g, '').padStart(6, '0').slice(-6) || patchSearch))

  if (loading) return <Loader2 className="mx-auto h-8 w-8 animate-spin text-accent" />

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text">Scene Explorer</h1>
        <p className="mt-1 text-sm text-muted">Prepared QA dataset — reference previews, not model inference.</p>
      </div>

      {err && <p className="text-sm text-error">{err}</p>}

      <div className="grid gap-6 xl:grid-cols-[300px_1fr]">
        <div className="space-y-3 rounded-xl border border-border-slate bg-surface p-4 shadow-glow">
          <input
            className="w-full rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs"
            placeholder="Filter scenes…"
            value={sceneFilter}
            onChange={(e) => setSceneFilter(e.target.value)}
          />
          <div className="max-h-48 overflow-y-auto space-y-1">
            {filteredScenes.map((s) => (
              <button
                key={s.scene_id}
                type="button"
                onClick={() => setSelection(s.scene_id, patchId)}
                className={`w-full text-left rounded-lg px-2 py-2 text-[10px] font-mono transition-colors ${
                  s.scene_id === sceneId ? 'bg-accent/10 text-accent' : 'text-muted hover:bg-surface-elevated'
                }`}
              >
                <span className="block truncate">{s.scene_id}</span>
                <span className="text-muted-dim">{s.split_role} · {s.patches_after_qa} patches</span>
              </button>
            ))}
          </div>
          <input
            className="w-full rounded-lg border border-border-slate bg-midnight px-3 py-2 text-xs font-mono"
            placeholder="Patch search"
            value={patchSearch || patchId}
            onChange={(e) => setPatchSearch(e.target.value)}
            onBlur={() => {
              const d = patchSearch.replace(/\D/g, '')
              if (d) setSelection(sceneId, d.padStart(6, '0'))
            }}
          />
          <select
            className="w-full rounded-lg border border-border-slate bg-midnight text-xs font-mono"
            size={5}
            value={patchId}
            onChange={(e) => setSelection(sceneId, e.target.value)}
          >
            {(patchSearch ? filtered : patchOptions.slice(0, 150)).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <Link
            to={`/studio?scene=${encodeURIComponent(sceneId)}&patch=${encodeURIComponent(patchId)}`}
            className="block text-center rounded-lg border border-accent/40 py-2 text-xs font-medium text-accent hover:bg-accent/10"
          >
            Open in Processing Studio
          </Link>
        </div>

        <div className="space-y-4">
          <div className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/5 px-4 py-3 text-xs text-warning">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {String(meta?.preview_note ?? 'Reference patch preview only.')}
          </div>
          {meta && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <Chip label="Valid px" value={String(meta.valid_pixels)} />
              <Chip label="Coverage" value={`${((meta.valid_fraction as number) * 100).toFixed(1)}%`} />
              <Chip label="Size" value={`${meta.width}×${meta.height}`} />
              <Chip
                label="Thermal μ"
                value={`${(meta.normalization as { thermal: { mean: number } }).thermal.mean.toFixed(2)} K`}
              />
            </div>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            <ImageViewer src={previewUrl(sceneId, patchId, 'thermal_colormap')} alt="Thermal" />
            <ImageViewer src={previewUrl(sceneId, patchId, 'rgb')} alt="RGB" />
          </div>
        </div>
      </div>
    </div>
  )
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border-slate bg-surface-elevated px-3 py-2">
      <p className="text-[9px] uppercase text-muted-dim">{label}</p>
      <p className="font-mono text-text mt-0.5">{value}</p>
    </div>
  )
}
