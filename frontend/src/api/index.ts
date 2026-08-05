import type {
  Scene,
  SceneFilters,
  DashboardStats,
  ActivityItem,
  SystemHealth,
  PreprocessingStep,
  TrainingMetrics,
  EvaluationMetrics,
  OutputItem,
  ExportHistoryItem,
  AppSettings,
  PipelineNodeDetail,
} from '@/types'

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms))

export async function fetchScenes(filters?: Partial<SceneFilters>): Promise<Scene[]> {
  await delay(400)
  const { scenes } = await import('@/data/mockData')
  let result = [...scenes]
  if (filters?.search) {
    const q = filters.search.toLowerCase()
    result = result.filter(
      (s) => s.sceneId.toLowerCase().includes(q) || s.location.toLowerCase().includes(q)
    )
  }
  if (filters?.satellite && filters.satellite !== 'all') {
    result = result.filter((s) => s.satellite === filters.satellite)
  }
  if (filters?.status && filters.status !== 'all') {
    result = result.filter((s) => s.status === filters.status)
  }
  if (filters?.cloudMax !== undefined && filters.cloudMax < 100) {
    result = result.filter((s) => s.cloudCover <= filters.cloudMax!)
  }
  return result
}

export async function fetchScene(id: string): Promise<Scene | null> {
  await delay(300)
  const { scenes } = await import('@/data/mockData')
  return scenes.find((s) => s.id === id) ?? null
}

export async function fetchDashboard(): Promise<{
  stats: DashboardStats
  activities: ActivityItem[]
  health: SystemHealth[]
}> {
  await delay(350)
  const { dashboardStats, activities, systemHealth } = await import('@/data/mockData')
  return { stats: dashboardStats, activities, health: systemHealth }
}

export async function fetchPreprocessingSteps(): Promise<PreprocessingStep[]> {
  await delay(300)
  const { preprocessingSteps } = await import('@/data/mockData')
  return preprocessingSteps
}

export async function fetchTrainingMetrics(): Promise<TrainingMetrics> {
  await delay(300)
  const { trainingMetrics } = await import('@/data/mockData')
  return trainingMetrics
}

export async function fetchEvaluation(sceneId?: string): Promise<EvaluationMetrics[]> {
  await delay(300)
  const { evaluationMetrics } = await import('@/data/mockData')
  if (sceneId) return evaluationMetrics.filter((m) => m.sceneId === sceneId)
  return evaluationMetrics
}

export async function fetchOutputs(): Promise<OutputItem[]> {
  await delay(300)
  const { outputs } = await import('@/data/mockData')
  return outputs
}

export async function fetchExportHistory(): Promise<ExportHistoryItem[]> {
  await delay(200)
  const { exportHistory } = await import('@/data/mockData')
  return exportHistory
}

export async function fetchSettings(): Promise<AppSettings> {
  await delay(200)
  const { defaultSettings } = await import('@/data/mockData')
  return { ...defaultSettings }
}

export async function fetchPipelineNodes(): Promise<PipelineNodeDetail[]> {
  await delay(250)
  const { pipelineNodes } = await import('@/data/mockData')
  return pipelineNodes
}

export async function fetchPreprocessingLogs(): Promise<string[]> {
  await delay(200)
  const { preprocessingLogs } = await import('@/data/mockData')
  return preprocessingLogs
}

export async function importExternalDataset(_file: File): Promise<{ success: boolean; message: string }> {
  await delay(1500)
  return { success: true, message: 'GeoTIFF queued for validation and catalog indexing' }
}
