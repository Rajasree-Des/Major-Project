import { Link, useLocation } from 'react-router-dom'
import { useHealth } from '@/hooks/useHealth'
import { useSelection } from '@/context/SelectionContext'
import { HelpCircle } from 'lucide-react'

const TITLES: Record<string, string> = {
  '/': 'Command Center',
  '/explorer': 'Scene Explorer',
  '/studio': 'Processing Studio',
  '/run': 'Processing Studio',
  '/results': 'Results Lab',
  '/colorization': 'Colorization Studio',
  '/history': 'Experiment History',
  '/about': 'Research & About',
}

export function TopBar() {
  const { pathname } = useLocation()
  const title = TITLES[pathname] ?? 'ThermaScope'
  const { health, connected, error } = useHealth()
  const { sceneId, patchId } = useSelection()

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-4 border-b border-border-slate bg-charcoal/95 px-6 backdrop-blur-md">
      <div>
        <p className="text-[10px] uppercase tracking-widest text-muted-dim">ThermaScope</p>
        <h1 className="text-sm font-semibold text-text">{title}</h1>
      </div>

      <div className="hidden md:flex items-center gap-2 text-xs font-mono text-muted">
        <span className="truncate max-w-[200px]">{sceneId.split('_').slice(-1)[0]}</span>
        <span className="text-muted-dim">/</span>
        <span>{patchId}</span>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`h-2 w-2 rounded-full ${connected ? 'bg-success shadow-[0_0_8px_rgba(52,211,153,0.6)]' : 'bg-error'}`}
            title={connected ? 'API connected' : error ?? 'Disconnected'}
          />
          <span className="text-muted">{connected ? 'API' : 'Offline'}</span>
          {health?.inference_busy && (
            <span className="rounded-md bg-accent/10 px-2 py-0.5 text-accent text-[10px] font-medium uppercase">
              Inference
            </span>
          )}
        </div>
        <Link to="/about" className="text-muted hover:text-accent transition-colors" title="About">
          <HelpCircle className="h-4 w-4" />
        </Link>
      </div>
    </header>
  )
}
