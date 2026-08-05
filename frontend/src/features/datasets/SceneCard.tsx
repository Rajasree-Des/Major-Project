import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { MoreHorizontal, Cloud, Satellite, ArrowUpRight } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/shared/StatusBadge'
import { ScenePreview } from '@/components/shared/ScenePreview'
import { ContextMenu } from '@/components/shared/ContextMenu'
import { formatDate, formatBytes } from '@/lib/utils'
import type { Scene } from '@/types'

interface SceneCardProps {
  scene: Scene
  index?: number
}

export function SceneCard({ scene, index = 0 }: SceneCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
    >
      <ContextMenu
        items={[
          { label: 'Open Scene', onClick: () => window.location.href = `/scenes/${scene.id}` },
          { label: 'Preprocess', onClick: () => {} },
          { label: 'Export GeoTIFF', onClick: () => {} },
          { label: 'Delete', danger: true, onClick: () => {} },
        ]}
      >
        <Card className="group overflow-hidden transition-all duration-300 hover:border-border-subtle hover:shadow-elevated">
          <ScenePreview gradient={scene.previewGradient} label={scene.thermalBand.name} />

          <div className="p-5 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="font-mono text-[11px] text-muted-dim truncate tracking-wide">{scene.sceneId}</p>
                <p className="text-sm font-medium text-foreground mt-1 truncate">{scene.location}</p>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <StatusBadge status={scene.status} />
                <button className="h-7 w-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-surface opacity-0 group-hover:opacity-100 transition-all">
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-muted">
              <span className="flex items-center gap-1.5">
                <Satellite className="h-3 w-3" />{scene.satellite}
              </span>
              <span className="flex items-center gap-1.5">
                <Cloud className="h-3 w-3" />{scene.cloudCover}%
              </span>
              <span>{formatDate(scene.acquisitionDate)}</span>
            </div>

            <div className="flex flex-wrap gap-1.5">
              <span className="inline-flex items-center rounded-md bg-surface border border-border px-2 py-0.5 text-[10px] font-medium text-muted uppercase tracking-wide">
                {scene.thermalBand.name}
              </span>
              {scene.rgbBands.map((b) => (
                <span key={b.name} className="inline-flex items-center rounded-md bg-surface border border-border px-2 py-0.5 text-[10px] font-medium text-muted-dim uppercase tracking-wide">
                  {b.name}
                </span>
              ))}
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-border">
              <span className="text-xs text-muted-dim">{formatBytes(scene.downloadSizeBytes)}</span>
              <Link to={`/scenes/${scene.id}`}>
                <Button size="sm" variant="secondary" className="gap-1.5">
                  Open Scene <ArrowUpRight className="h-3 w-3" />
                </Button>
              </Link>
            </div>
          </div>
        </Card>
      </ContextMenu>
    </motion.div>
  )
}
