import { Link } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'
import { fetchScenes } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { StatusBadge } from '@/components/shared/StatusBadge'
import { SceneCardSkeleton } from '@/components/ui/skeleton'
import { formatDate, formatBytes } from '@/lib/utils'
import { FadeIn } from '@/components/shared/PageTransition'

export function ScenesPage() {
  const { data: scenes, loading } = useAsyncData(() => fetchScenes())

  return (
    <div className="space-y-8">
      <FadeIn>
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Scenes</h2>
          <p className="text-sm text-muted mt-1">Dense catalog view of all acquired satellite scenes</p>
        </div>
      </FadeIn>

      {loading ? (
        <div className="space-y-2">{Array.from({ length: 6 }).map((_, i) => <SceneCardSkeleton key={i} />)}</div>
      ) : (
        <FadeIn delay={0.05}>
          <div className="rounded-xl border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-surface">
                  {['Scene ID', 'Location', 'Satellite', 'Date', 'Cloud', 'Size', 'Status', ''].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-[11px] font-medium text-muted-dim uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {scenes?.map((scene) => (
                  <tr key={scene.id} className="border-b border-border last:border-0 hover:bg-card transition-colors group">
                    <td className="px-4 py-3 font-mono text-xs text-muted">{scene.sceneId.slice(0, 28)}…</td>
                    <td className="px-4 py-3 text-foreground">{scene.location}</td>
                    <td className="px-4 py-3 text-muted">{scene.satellite}</td>
                    <td className="px-4 py-3 text-muted">{formatDate(scene.acquisitionDate)}</td>
                    <td className="px-4 py-3 text-muted">{scene.cloudCover}%</td>
                    <td className="px-4 py-3 text-muted-dim">{formatBytes(scene.downloadSizeBytes)}</td>
                    <td className="px-4 py-3"><StatusBadge status={scene.status} /></td>
                    <td className="px-4 py-3">
                      <Link to={`/scenes/${scene.id}`} className="opacity-0 group-hover:opacity-100 transition-opacity text-primary">
                        <ArrowUpRight className="h-4 w-4" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </FadeIn>
      )}
    </div>
  )
}
