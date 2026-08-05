import { Download } from 'lucide-react'
import { fetchOutputs, fetchExportHistory } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScenePreview } from '@/components/shared/ScenePreview'
import { FadeIn } from '@/components/shared/PageTransition'
import { formatBytes, formatDateTime } from '@/lib/utils'

export function OutputsPage() {
  const { data: outputs, loading: loadingOutputs } = useAsyncData(() => fetchOutputs())
  const { data: history, loading: loadingHistory } = useAsyncData(() => fetchExportHistory())

  return (
    <div className="space-y-8">
      <FadeIn>
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Outputs</h2>
          <p className="text-sm text-muted mt-1">Generated GeoTIFF products and export history</p>
        </div>
      </FadeIn>

      {loadingOutputs ? (
        <div className="grid gap-6 md:grid-cols-2 animate-pulse">
          {Array.from({ length: 4 }).map((_, i) => <div key={i} className="h-64 rounded-xl border border-border" />)}
        </div>
      ) : (
        <FadeIn delay={0.05}>
          <div className="grid gap-6 md:grid-cols-2">
            {outputs?.map((output) => (
              <Card key={output.id} className="overflow-hidden group hover:border-border-subtle transition-all hover:shadow-elevated">
                <ScenePreview gradient={output.thumbnailGradient} label={output.format} />
                <div className="p-6 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-medium">{output.title}</h3>
                      <p className="font-mono text-[10px] text-muted-dim mt-1 truncate">{output.sceneId}</p>
                    </div>
                    <Badge variant="outline">{output.type}</Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-muted">
                    <span>{output.resolution}</span>
                    <span>{formatBytes(output.sizeBytes)}</span>
                    <span>{output.crs}</span>
                    <span>{formatDateTime(output.createdAt)}</span>
                  </div>
                  <p className="text-[11px] text-muted-dim">Bands: {output.bands.join(', ')}</p>
                  <Button className="w-full" size="sm"><Download className="h-3.5 w-3.5" /> Download</Button>
                </div>
              </Card>
            ))}
          </div>
        </FadeIn>
      )}

      <FadeIn delay={0.1}>
        <Card>
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold">Export History</h3>
          </div>
          {loadingHistory ? (
            <div className="p-6 h-32 animate-pulse" />
          ) : (
            <div className="divide-y divide-border">
              {history?.map((item) => (
                <div key={item.id} className="flex items-center justify-between px-6 py-4 hover:bg-surface/50 transition-colors">
                  <div className="min-w-0">
                    <p className="text-sm font-mono truncate">{item.filename}</p>
                    <p className="text-xs text-muted-dim mt-0.5">{formatDateTime(item.exportedAt)} · {item.format}</p>
                  </div>
                  <div className="flex items-center gap-4 shrink-0">
                    <span className="text-xs text-muted">{item.sizeBytes > 0 ? formatBytes(item.sizeBytes) : '—'}</span>
                    <Badge variant={item.status === 'completed' ? 'success' : 'danger'}>{item.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </FadeIn>
    </div>
  )
}
