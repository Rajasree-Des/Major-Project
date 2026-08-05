import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppLayout } from '@/components/layout/AppLayout'
import { DashboardPage } from '@/features/dashboard/DashboardPage'
import { DatasetsPage } from '@/features/datasets/DatasetsPage'
import { ScenesPage } from '@/features/scenes/ScenesPage'
import { SceneDetailsPage } from '@/features/scenes/SceneDetailsPage'
import { PreprocessingPage } from '@/features/preprocessing/PreprocessingPage'
import { ModelPipelinePage } from '@/features/models/ModelPipelinePage'
import { TrainingPage } from '@/features/training/TrainingPage'
import { EvaluationPage } from '@/features/evaluation/EvaluationPage'
import { OutputsPage } from '@/features/outputs/OutputsPage'
import { SettingsPage } from '@/features/settings/SettingsPage'

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="scenes" element={<ScenesPage />} />
          <Route path="scenes/:id" element={<SceneDetailsPage />} />
          <Route path="preprocessing" element={<PreprocessingPage />} />
          <Route path="models" element={<ModelPipelinePage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="outputs" element={<OutputsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          {/* Legacy redirects */}
          <Route path="pipeline" element={<Navigate to="/models" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
