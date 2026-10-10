const API_BASE = import.meta.env.VITE_API_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

export function artifactUrl(rootKey: string, relativePath: string) {
  const clean = relativePath.replace(/\\/g, '/').replace(/^\/+/, '')
  return `${API_BASE}/api/artifacts/${rootKey}/${clean}`
}

export function previewUrl(sceneId: string, patchId: string, kind: 'thermal_colormap' | 'rgb') {
  return `${API_BASE}/api/scenes/${encodeURIComponent(sceneId)}/patches/${encodeURIComponent(patchId)}/preview/${kind}`
}

export type Health = {
  status: string
  scene_count: number
  pretrained_checkpoint_available: boolean
  frozen_refinement_beta: number
  device: string
  inference_busy?: boolean
  missing_requirements?: string[]
}

export type SceneSummary = {
  scene_id: string
  patches_after_qa?: number
  split_role?: string
  region?: string
  sensor?: string
  acquisition_date?: string
  thermal_norm?: { mean: number; std: number }
}

export type Overview = {
  project_name: string
  tagline: string
  subtitle?: string
  scenes_prepared: number
  total_patches?: number
  demo_scene_id?: string
  demo_patch_id?: string
  demo_patch_available?: boolean
  scenes: SceneSummary[]
  checkpoint: { variant: string; available: boolean }
  frozen_config_summary: Record<string, unknown>
  latest_run: Record<string, unknown> | null
  historical_runs: Record<string, unknown>[]
  frozen_evaluations_available?: boolean
}

export type FrozenResults = {
  experiment: string
  role: string
  locked: boolean
  disclaimer: string
  scene_ids: string[]
  per_scene: Array<{
    scene_id: string
    metrics: Record<string, Record<string, number>>
  }>
  aggregate?: Record<string, unknown>
  methods: string[]
}

export type RunManifest = {
  run_id: string
  status: string
  stage?: string
  output_root_key?: string
  runtime_s?: number
  patch_ids?: string[]
  pipeline_version?: string
  stage_artifacts?: Record<string, string>
  patches?: Array<{
    patch_id: string
    scene_id: string
    colorization_outputs?: Record<string, string>
    thermal_kelvin_file?: string
    valid_mask_file?: string
    valid_pixels?: number
  }>
  sr_comparisons?: Array<{
    patch_id: string
    metrics_vs_hr_reference: Record<string, { rmse_kelvin?: number; mae_kelvin?: number; psnr?: number; ssim?: number } | null>
    metrics_note?: string
  }>
  logs?: string[]
  progress_message?: string
  error?: string
}

export const api = {
  health: () => request<Health>('/api/health'),
  overview: () => request<Overview>('/api/overview'),
  scenes: () => request<SceneSummary[]>('/api/scenes'),
  patches: (sceneId: string, limit = 500, offset = 0) =>
    request<{ total: number; patch_ids: string[] }>(
      `/api/scenes/${encodeURIComponent(sceneId)}/patches?limit=${limit}&offset=${offset}`
    ),
  patchMeta: (sceneId: string, patchId: string) =>
    request<Record<string, unknown>>(`/api/scenes/${encodeURIComponent(sceneId)}/patches/${encodeURIComponent(patchId)}`),
  frozenResults: (experiment: 'multiscene' | 'physics', role: 'validation' | 'test_locked') =>
    request<FrozenResults>(`/api/results/frozen?experiment=${experiment}&role=${role}`),
  frozenEval: () => request<Array<{ id: string; excerpt_lines: string[]; is_locked_test: boolean }>>('/api/frozen-evaluations'),
  history: () => request<RunManifest[]>('/api/history'),
  runs: () => request<RunManifest[]>('/api/runs'),
  run: (runId: string) => request<RunManifest>(`/api/runs/${encodeURIComponent(runId)}`),
  submitJob: (body: {
    mode: 'single' | 'validation_sample'
    scene_id?: string
    patch_id?: string
    num_patches?: number
    seed?: number
  }) => request<{ job_id: string; status: string }>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  job: (jobId: string) => request<RunManifest>(`/api/jobs/${encodeURIComponent(jobId)}`),
  jobLogs: (jobId: string) => request<{ lines: string[] }>(`/api/jobs/${encodeURIComponent(jobId)}/logs`),
  jobArtifacts: (jobId: string) =>
    request<{ artifacts: Array<{ name: string; relative_path: string; root_key: string }> }>(
      `/api/jobs/${encodeURIComponent(jobId)}/artifacts`
    ),
}
