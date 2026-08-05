import { useEffect, useState, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  Database,
  Layers,
  Wrench,
  GitBranch,
  Brain,
  BarChart3,
  Image,
  Settings,
  Search,
  FileUp,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navCommands = [
  { id: 'dashboard', label: 'Dashboard', path: '/', icon: LayoutDashboard, group: 'Navigation' },
  { id: 'datasets', label: 'Datasets', path: '/datasets', icon: Database, group: 'Navigation' },
  { id: 'scenes', label: 'Scenes', path: '/scenes', icon: Layers, group: 'Navigation' },
  { id: 'preprocessing', label: 'Preprocessing', path: '/preprocessing', icon: Wrench, group: 'Navigation' },
  { id: 'models', label: 'Models', path: '/models', icon: GitBranch, group: 'Navigation' },
  { id: 'training', label: 'Training', path: '/training', icon: Brain, group: 'Navigation' },
  { id: 'evaluation', label: 'Evaluation', path: '/evaluation', icon: BarChart3, group: 'Navigation' },
  { id: 'outputs', label: 'Outputs', path: '/outputs', icon: Image, group: 'Navigation' },
  { id: 'settings', label: 'Settings', path: '/settings', icon: Settings, group: 'Navigation' },
  { id: 'import', label: 'Import External Dataset', path: '/datasets?import=true', icon: FileUp, group: 'Actions' },
]

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const filtered = navCommands.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase())
  )

  const execute = useCallback(
    (cmd: (typeof navCommands)[0]) => {
      navigate(cmd.path)
      onClose()
      setQuery('')
    },
    [navigate, onClose]
  )

  useEffect(() => {
    if (open) {
      setQuery('')
      setSelected(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        if (open) onClose()
      }
      if (!open) return
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelected((s) => Math.min(s + 1, filtered.length - 1))
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelected((s) => Math.max(s - 1, 0))
      }
      if (e.key === 'Enter' && filtered[selected]) {
        execute(filtered[selected])
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, filtered, selected, execute, onClose])

  const groups = [...new Set(filtered.map((c) => c.group))]

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/50 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.98, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: -20 }}
            transition={{ duration: 0.15 }}
            className="fixed left-1/2 top-[20%] z-[101] w-full max-w-lg -translate-x-1/2 rounded-2xl border border-border bg-card shadow-elevated overflow-hidden"
          >
            <div className="flex items-center gap-3 border-b border-border px-4">
              <Search className="h-4 w-4 text-muted shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => { setQuery(e.target.value); setSelected(0) }}
                placeholder="Search commands, pages, scenes..."
                className="flex-1 h-12 bg-transparent text-sm text-foreground placeholder:text-muted-dim outline-none"
              />
              <kbd className="hidden sm:inline text-[10px] text-muted-dim bg-surface border border-border rounded px-1.5 py-0.5">ESC</kbd>
            </div>
            <div className="max-h-72 overflow-y-auto p-2">
              {filtered.length === 0 ? (
                <p className="text-sm text-muted text-center py-8">No results found</p>
              ) : (
                groups.map((group) => (
                  <div key={group} className="mb-2">
                    <p className="text-[10px] font-medium text-muted-dim uppercase tracking-wider px-2 py-1.5">{group}</p>
                    {filtered.filter((c) => c.group === group).map((cmd) => {
                      const idx = filtered.indexOf(cmd)
                      const Icon = cmd.icon
                      return (
                        <button
                          key={cmd.id}
                          onClick={() => execute(cmd)}
                          className={cn(
                            'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors',
                            idx === selected ? 'bg-primary/10 text-foreground' : 'text-muted hover:bg-surface hover:text-foreground'
                          )}
                        >
                          <Icon className="h-4 w-4 shrink-0" />
                          <span>{cmd.label}</span>
                        </button>
                      )
                    })}
                  </div>
                ))
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export function useCommandPalette() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return { open, setOpen, toggle: () => setOpen((o) => !o), close: () => setOpen(false) }
}
