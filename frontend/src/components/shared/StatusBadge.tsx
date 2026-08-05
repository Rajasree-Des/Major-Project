import type { SceneStatus, ProcessingStepStatus } from '@/types'
import { Badge } from '@/components/ui/badge'

const sceneMap: Record<SceneStatus, { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'secondary' | 'outline' }> = {
  downloaded: { label: 'Downloaded', variant: 'secondary' },
  preprocessing: { label: 'Preprocessing', variant: 'warning' },
  preprocessed: { label: 'Preprocessed', variant: 'default' },
  processing: { label: 'Processing', variant: 'warning' },
  processed: { label: 'Processed', variant: 'success' },
  failed: { label: 'Failed', variant: 'danger' },
}

const stepMap: Record<ProcessingStepStatus, { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'secondary' | 'outline' }> = {
  completed: { label: 'Completed', variant: 'success' },
  running: { label: 'Running', variant: 'warning' },
  pending: { label: 'Pending', variant: 'secondary' },
  waiting: { label: 'Waiting', variant: 'outline' },
  failed: { label: 'Failed', variant: 'danger' },
}

export function StatusBadge({ status, type = 'scene' }: { status: SceneStatus | ProcessingStepStatus; type?: 'scene' | 'step' }) {
  const config = type === 'scene' ? sceneMap[status as SceneStatus] : stepMap[status as ProcessingStepStatus]
  return <Badge variant={config.variant}>{config.label}</Badge>
}
