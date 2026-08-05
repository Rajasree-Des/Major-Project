export type SceneStatus =
  | 'downloaded'
  | 'preprocessing'
  | 'preprocessed'
  | 'processing'
  | 'processed'
  | 'failed'

export type ProcessingStepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'waiting'

export type BandType = 'thermal' | 'blue' | 'green' | 'red'

export interface BoundingBox {
  west: number
  south: number
  east: number
  north: number
}

export interface BandInfo {
  name: string
  resolution: string
  wavelength?: string
  sizeBytes: number
}

export interface Scene {
  id: string
  sceneId: string
  satellite: 'Landsat-8' | 'Landsat-9'
  acquisitionDate: string
  cloudCover: number
  boundingBox: BoundingBox
  pathRow: string
  thermalBand: BandInfo
  rgbBands: BandInfo[]
  status: SceneStatus
  downloadSizeBytes: number
  centerLat: number
  centerLon: number
  location: string
  processingProgress: number
  sentinelRgbAvailable: boolean
  previewGradient: string
}

export interface ActivityItem {
  id: string
  type: 'download' | 'preprocess' | 'inference' | 'export' | 'training' | 'system'
  message: string
  timestamp: string
  status: 'success' | 'warning' | 'error' | 'info'
  sceneId?: string
}

export interface SystemHealth {
  component: string
  status: 'healthy' | 'degraded' | 'offline'
  latencyMs?: number
  message: string
}

export interface PipelineStage {
  id: string
  name: string
  status: ProcessingStepStatus
}

export interface DashboardStats {
  downloadedScenes: number
  processedScenes: number
  storageUsedBytes: number
  storageTotalBytes: number
  trainingStatus: 'idle' | 'running' | 'completed' | 'paused'
  currentModel: string
  modelVersion: string
  gpuAvailable: boolean
  gpuName: string
  lastProcessedAt: string
  pipelineStages: PipelineStage[]
}

export interface PreprocessingStep {
  id: string
  name: string
  description: string
  status: ProcessingStepStatus
  progress: number
  duration?: string
  logs?: string[]
}

export interface TrainingMetrics {
  epoch: number
  totalEpochs: number
  trainLoss: number
  valLoss: number
  learningRate: number
  gpuUsage: number
  gpuMemoryUsed: number
  gpuMemoryTotal: number
  modelVersion: string
  datasetName: string
  checkpoint: string
  batchSize: number
  samplesProcessed: number
  totalSamples: number
  eta: string
  bestValLoss: number
  stepsPerEpoch: number
  currentStep: number
}

export interface EvaluationMetrics {
  psnr: number
  ssim: number
  lpips: number
  rmse: number
  sceneId: string
  modelVersion: string
  evaluatedAt: string
}

export interface MetricExplanation {
  name: string
  value: number
  unit: string
  description: string
  threshold: string
  pass: boolean
}

export interface OutputItem {
  id: string
  sceneId: string
  title: string
  type: 'super-resolution' | 'colorization' | 'combined'
  format: 'GeoTIFF' | 'PNG' | 'COG'
  resolution: string
  sizeBytes: number
  createdAt: string
  crs: string
  bands: string[]
  thumbnailGradient: string
}

export interface ExportHistoryItem {
  id: string
  filename: string
  sceneId: string
  format: string
  sizeBytes: number
  exportedAt: string
  status: 'completed' | 'failed'
}

export interface AppSettings {
  datasetPath: string
  modelCheckpoint: string
  modelSelection: string
  gpuDevice: string
  patchSize: number
  scaleFactor: number
  batchSize: number
  numWorkers: number
  outputFormat: string
  theme: 'dark' | 'light'
}

export interface PipelineNodeDetail {
  id: string
  label: string
  purpose: string
  input: string
  output: string
  model: string
  parameters: Record<string, string | number | boolean>
  status: ProcessingStepStatus
}

export interface SceneFilters {
  search: string
  satellite: string
  cloudMax: number
  dateFrom: string
  dateTo: string
  status: string
}

export interface CommandItem {
  id: string
  label: string
  description?: string
  icon?: string
  shortcut?: string
  action: () => void
  group: string
}
