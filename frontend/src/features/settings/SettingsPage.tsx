import { useState } from 'react'
import { Save } from 'lucide-react'
import { fetchSettings } from '@/api'
import { useAsyncData } from '@/hooks/useAsyncData'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { FadeIn } from '@/components/shared/PageTransition'
import type { AppSettings } from '@/types'

export function SettingsPage() {
  const { data: initial } = useAsyncData(() => fetchSettings())
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [saved, setSaved] = useState(false)

  const current = settings ?? initial

  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => ({ ...(prev ?? initial!), [key]: value }))
    setSaved(false)
  }

  if (!current) return <div className="h-64 rounded-2xl border border-border animate-pulse" />

  return (
    <div className="space-y-8 max-w-2xl">
      <FadeIn>
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Settings</h2>
            <p className="text-sm text-muted mt-1">System configuration and preferences</p>
          </div>
          <Button size="sm" onClick={() => setSaved(true)}>
            <Save className="h-3.5 w-3.5" />{saved ? 'Saved' : 'Save'}
          </Button>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <Card className="p-6 space-y-5">
          <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider">Storage</h3>
          <div className="space-y-2">
            <Label htmlFor="dataset">Dataset Directory</Label>
            <Input id="dataset" value={current.datasetPath} onChange={(e) => update('datasetPath', e.target.value)} />
          </div>
        </Card>
      </FadeIn>

      <FadeIn delay={0.08}>
        <Card className="p-6 space-y-5">
          <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider">Model</h3>
          <div className="space-y-2">
            <Label htmlFor="model">Model Selection</Label>
            <Select id="model" value={current.modelSelection} onChange={(e) => update('modelSelection', e.target.value)}>
              <option value="thermal-swinir-v2.1.0">thermal-swinir-v2.1.0</option>
              <option value="thermal-swinir-v2.0.0">thermal-swinir-v2.0.0</option>
              <option value="thermal-swinir-v1.0.0">thermal-swinir-v1.0.0</option>
            </Select>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="patch">Patch Size</Label>
              <Select id="patch" value={current.patchSize} onChange={(e) => update('patchSize', Number(e.target.value))}>
                <option value={64}>64</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="scale">Scale Factor</Label>
              <Select id="scale" value={current.scaleFactor} onChange={(e) => update('scaleFactor', Number(e.target.value))}>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
                <option value={8}>8×</option>
              </Select>
            </div>
          </div>
        </Card>
      </FadeIn>

      <FadeIn delay={0.11}>
        <Card className="p-6 space-y-5">
          <h3 className="text-xs font-medium text-muted-dim uppercase tracking-wider">Compute</h3>
          <div className="space-y-2">
            <Label htmlFor="gpu">GPU Device</Label>
            <Select id="gpu" value={current.gpuDevice} onChange={(e) => update('gpuDevice', e.target.value)}>
              <option value="cuda:0 — NVIDIA GeForce RTX 4090">cuda:0 — RTX 4090</option>
              <option value="cpu">CPU Only</option>
            </Select>
          </div>
        </Card>
      </FadeIn>

      <FadeIn delay={0.14}>
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Dark Theme</p>
              <p className="text-xs text-muted mt-0.5">Optimized for geospatial analysis workflows</p>
            </div>
            <Switch
              checked={current.theme === 'dark'}
              onCheckedChange={(v) => update('theme', v ? 'dark' : 'light')}
            />
          </div>
        </Card>
      </FadeIn>
    </div>
  )
}
