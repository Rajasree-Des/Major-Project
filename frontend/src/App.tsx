import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppLayout } from '@/components/layout/AppLayout'
import { OverviewPage } from '@/features/thermascope/OverviewPage'
import { SceneExplorerPage } from '@/features/thermascope/SceneExplorerPage'
import { ProcessingStudioPage } from '@/features/thermascope/ProcessingStudioPage'
import { ResultsPage } from '@/features/thermascope/ResultsPage'
import { ColorizationStudioPage } from '@/features/thermascope/ColorizationStudioPage'
import { HistoryPage } from '@/features/thermascope/HistoryPage'
import { AboutPage } from '@/features/thermascope/AboutPage'

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="explorer" element={<SceneExplorerPage />} />
          <Route path="studio" element={<ProcessingStudioPage />} />
          <Route path="run" element={<Navigate to="/studio" replace />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="colorization" element={<ColorizationStudioPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="about" element={<AboutPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
