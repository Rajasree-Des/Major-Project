import { Search, Command } from 'lucide-react'
import { useLocation } from 'react-router-dom'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/datasets': 'Datasets',
  '/scenes': 'Scenes',
  '/preprocessing': 'Preprocessing',
  '/models': 'Models',
  '/training': 'Training',
  '/evaluation': 'Evaluation',
  '/outputs': 'Outputs',
  '/settings': 'Settings',
}

interface CommandBarProps {
  onOpenCommand: () => void
}

export function CommandBar({ onOpenCommand }: CommandBarProps) {
  const location = useLocation()
  const title = Object.entries(pageTitles).find(([path]) =>
    path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)
  )?.[1] ?? 'ThermaScope AI'

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-background/80 px-6 backdrop-blur-xl">
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-medium text-foreground">{title}</h1>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onOpenCommand}
          className="flex items-center gap-3 h-8 px-3 rounded-xl border border-border bg-surface text-muted hover:text-foreground hover:border-border-subtle transition-all text-sm"
        >
          <Search className="h-3.5 w-3.5" />
          <span className="hidden sm:inline text-muted-dim">Search...</span>
          <kbd className="hidden sm:inline-flex items-center gap-0.5 text-[10px] text-muted-dim bg-card border border-border rounded px-1.5 py-0.5 ml-4">
            <Command className="h-2.5 w-2.5" />K
          </kbd>
        </button>
        <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-[10px] font-semibold text-primary">
          IS
        </div>
      </div>
    </header>
  )
}
