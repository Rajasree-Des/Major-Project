import { useState } from 'react'
import { motion } from 'framer-motion'
import { fetchPipelineNodes } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { SidePanel } from '@/components/shared/SidePanel'
import { Badge } from '@/components/ui/badge'
import { FadeIn } from '@/components/shared/PageTransition'
import { cn } from '@/lib/utils'
import type { PipelineNodeDetail } from '@/types'

function FlowArrow({ className }: { className?: string }) {
  return (
    <div className={cn('flex justify-center py-2', className)}>
      <svg width="20" height="20" viewBox="0 0 20 20" className="text-muted-dim">
        <path d="M10 4v12M6 12l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

function PipelineCard({
  node,
  onClick,
  variant = 'default',
}: {
  node: PipelineNodeDetail
  onClick: () => void
  variant?: 'default' | 'decision' | 'branch'
}) {
  return (
    <motion.button
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={cn(
        'w-full max-w-sm rounded-xl border p-5 text-left transition-shadow hover:shadow-elevated',
        variant === 'decision' ? 'border-warning/30 bg-warning/5'
        : variant === 'branch' ? 'border-primary/20 bg-primary/5'
        : 'border-border bg-card hover:border-border-subtle',
        node.status === 'running' && 'ring-1 ring-primary/30'
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <Badge variant={node.status === 'completed' ? 'success' : node.status === 'running' ? 'warning' : 'outline'}>
          {node.status}
        </Badge>
      </div>
      <h3 className="text-sm font-semibold">{node.label}</h3>
      <p className="text-[11px] text-muted mt-1.5 leading-relaxed line-clamp-2">{node.purpose}</p>
    </motion.button>
  )
}

function NodePanel({ node }: { node: PipelineNodeDetail }) {
  return (
    <div className="space-y-6">
      <div>
        <Badge variant="outline" className="mb-3">{node.status}</Badge>
        <p className="text-sm text-muted leading-relaxed">{node.purpose}</p>
      </div>
      {[
        { label: 'Input', value: node.input },
        { label: 'Output', value: node.output },
        { label: 'Model', value: node.model },
      ].map(({ label, value }) => (
        <div key={label}>
          <p className="text-[10px] font-medium text-muted-dim uppercase tracking-wider mb-1">{label}</p>
          <p className="text-sm font-mono text-foreground">{value}</p>
        </div>
      ))}
      <div>
        <p className="text-[10px] font-medium text-muted-dim uppercase tracking-wider mb-2">Parameters</p>
        <div className="rounded-xl bg-surface border border-border p-4 space-y-2">
          {Object.entries(node.parameters).map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs">
              <span className="text-muted">{k}</span>
              <span className="font-mono">{String(v)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export function ModelPipelinePage() {
  const [selected, setSelected] = useState<PipelineNodeDetail | null>(null)
  const { data: nodes, loading } = useAsyncData(() => fetchPipelineNodes())

  if (loading || !nodes) return <div className="h-96 rounded-2xl border border-border animate-pulse" />

  const get = (id: string) => nodes.find((n) => n.id === id)!

  return (
    <div className="space-y-8">
      <FadeIn>
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Model Pipeline</h2>
          <p className="text-sm text-muted mt-1">SwinIR thermal super-resolution with adaptive colorization pathways</p>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="flex flex-col items-center py-4">
          <PipelineCard node={get('thermal-input')} onClick={() => setSelected(get('thermal-input'))} />
          <FlowArrow />
          <PipelineCard node={get('swinir')} onClick={() => setSelected(get('swinir'))} />
          <FlowArrow />
          <PipelineCard node={get('color-guidance')} onClick={() => setSelected(get('color-guidance'))} />
          <FlowArrow />
          <PipelineCard node={get('decision')} onClick={() => setSelected(get('decision'))} variant="decision" />

          <div className="my-2 rounded-lg bg-surface border border-border px-4 py-2">
            <p className="text-xs font-medium text-center">Sentinel RGB Available?</p>
          </div>

          <div className="flex gap-8 my-2">
            <div className="flex flex-col items-center">
              <Badge variant="success" className="mb-2">YES</Badge>
              <PipelineCard node={get('cross-modal')} onClick={() => setSelected(get('cross-modal'))} variant="branch" />
            </div>
            <div className="flex flex-col items-center">
              <Badge variant="secondary" className="mb-2">NO</Badge>
              <PipelineCard node={get('self-guided')} onClick={() => setSelected(get('self-guided'))} variant="branch" />
            </div>
          </div>

          <FlowArrow />
          <PipelineCard node={get('unified')} onClick={() => setSelected(get('unified'))} />
          <FlowArrow />
          <PipelineCard node={get('physics-refine')} onClick={() => setSelected(get('physics-refine'))} />
          <FlowArrow />
          <PipelineCard node={get('geotiff-output')} onClick={() => setSelected(get('geotiff-output'))} />
        </div>
      </FadeIn>

      <SidePanel
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected?.label ?? ''}
      >
        {selected && <NodePanel node={selected} />}
      </SidePanel>
    </div>
  )
}
