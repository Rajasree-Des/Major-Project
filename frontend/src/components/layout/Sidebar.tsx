import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Map,
  Cpu,
  Columns3,
  Palette,
  History,
  BookOpen,
  Thermometer,
  PanelLeftClose,
  PanelLeft,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useEffect, useState } from 'react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Command Center', end: true },
  { to: '/explorer', icon: Map, label: 'Scene Explorer' },
  { to: '/studio', icon: Cpu, label: 'Processing Studio' },
  { to: '/results', icon: Columns3, label: 'Results Lab' },
  { to: '/colorization', icon: Palette, label: 'Colorization Studio' },
  { to: '/history', icon: History, label: 'Experiment History' },
  { to: '/about', icon: BookOpen, label: 'Research & About' },
]

export function Sidebar() {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ts-sidebar') === 'collapsed')

  useEffect(() => {
    localStorage.setItem('ts-sidebar', collapsed ? 'collapsed' : 'expanded')
  }, [collapsed])

  const w = collapsed ? 'w-[72px]' : 'w-[248px]'

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border-slate bg-charcoal transition-[width] duration-200',
        w
      )}
    >
      <div className={cn('flex items-center gap-3 border-b border-border-slate px-4 py-4', collapsed && 'justify-center px-2')}>
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 ring-1 ring-accent/25">
          <Thermometer className="h-4 w-4 text-accent" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="text-sm font-bold tracking-tight text-text">THERMASCOPE</p>
            <p className="text-[9px] leading-tight text-muted uppercase tracking-wide">
              Thermal SR Studio
            </p>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {navItems.map(({ to, icon: Icon, label, end }) => {
          const active = end ? location.pathname === to : location.pathname.startsWith(to)
          return (
            <NavLink key={to} to={to} end={end} title={label} className="block">
              <div
                className={cn(
                  'relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                  collapsed && 'justify-center px-0',
                  active
                    ? 'bg-accent/10 text-accent shadow-[inset_0_0_0_1px_rgb(34_211_238/0.2)]'
                    : 'text-muted hover:bg-surface-elevated hover:text-text'
                )}
              >
                <Icon className="h-[18px] w-[18px] shrink-0" strokeWidth={1.75} />
                {!collapsed && <span className="truncate">{label}</span>}
              </div>
            </NavLink>
          )
        })}
      </nav>

      <div className="border-t border-border-slate p-2">
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="flex w-full items-center justify-center gap-2 rounded-lg py-2 text-xs text-muted hover:bg-surface-elevated hover:text-text"
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          {!collapsed && 'Collapse'}
        </button>
        {!collapsed && (
          <p className="px-2 pb-2 text-[10px] text-muted-dim">Prototype · CPU · QA v1</p>
        )}
      </div>
    </aside>
  )
}

export function sidebarWidth(collapsed: boolean) {
  return collapsed ? 72 : 248
}
