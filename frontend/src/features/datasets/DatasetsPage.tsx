import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Search, SlidersHorizontal, FileUp } from 'lucide-react'
import { fetchScenes } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { SceneCard } from '@/features/datasets/SceneCard'
import { ImportDatasetModal } from '@/features/datasets/ImportDatasetModal'
import { SceneCardSkeleton } from '@/components/ui/skeleton'
import { EmptyState } from '@/components/shared/EmptyState'
import { FadeIn } from '@/components/shared/PageTransition'
import { Database } from 'lucide-react'
import type { SceneFilters } from '@/types'

export function DatasetsPage() {
  const [searchParams] = useSearchParams()
  const [importOpen, setImportOpen] = useState(searchParams.get('import') === 'true')
  const [filters, setFilters] = useState<SceneFilters>({
    search: '',
    satellite: 'all',
    cloudMax: 100,
    dateFrom: '',
    dateTo: '',
    status: 'all',
  })

  const { data: scenes, loading } = useAsyncData(
    () => fetchScenes(filters),
    [filters.search, filters.satellite, filters.status, filters.cloudMax]
  )

  useEffect(() => {
    if (searchParams.get('import') === 'true') setImportOpen(true)
  }, [searchParams])

  return (
    <div className="space-y-8">
      <FadeIn>
        <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Dataset Explorer</h2>
            <p className="text-sm text-muted mt-1">
              Browse Landsat 8/9 Collection 2 Level-2 scenes from Planetary Computer
            </p>
          </div>
          <Button size="sm" variant="secondary" onClick={() => setImportOpen(true)}>
            <FileUp className="h-3.5 w-3.5" />
            Import External Dataset
          </Button>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-dim" />
            <Input
              placeholder="Search scene ID, location..."
              className="pl-9"
              value={filters.search}
              onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Select value={filters.satellite} onChange={(e) => setFilters((f) => ({ ...f, satellite: e.target.value }))} className="w-36">
              <option value="all">All Satellites</option>
              <option value="Landsat-8">Landsat-8</option>
              <option value="Landsat-9">Landsat-9</option>
            </Select>
            <Select value={String(filters.cloudMax)} onChange={(e) => setFilters((f) => ({ ...f, cloudMax: Number(e.target.value) }))} className="w-32">
              <option value="100">Any Cloud %</option>
              <option value="10">Cloud ≤ 10%</option>
              <option value="20">Cloud ≤ 20%</option>
              <option value="50">Cloud ≤ 50%</option>
            </Select>
            <Select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))} className="w-36">
              <option value="all">All Status</option>
              <option value="downloaded">Downloaded</option>
              <option value="preprocessing">Preprocessing</option>
              <option value="processed">Processed</option>
            </Select>
            <Button variant="outline" size="icon"><SlidersHorizontal className="h-4 w-4" /></Button>
          </div>
        </div>
      </FadeIn>

      {loading ? (
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => <SceneCardSkeleton key={i} />)}
        </div>
      ) : !scenes?.length ? (
        <EmptyState
          icon={Database}
          title="No scenes found"
          description="Adjust your filters or download scenes from Microsoft Planetary Computer."
        />
      ) : (
        <>
          <p className="text-xs text-muted-dim">{scenes.length} scenes</p>
          <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
            {scenes.map((scene, i) => (
              <SceneCard key={scene.id} scene={scene} index={i} />
            ))}
          </div>
        </>
      )}

      <ImportDatasetModal open={importOpen} onClose={() => setImportOpen(false)} />
    </div>
  )
}
