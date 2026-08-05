import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
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
  Orbit,
  HardDrive,
  Cpu,
} from 'lucide-react'
import { cn, formatBytes } from '@/lib/utils'
import { dashboardStats } from '@/data/mockData'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', end: true },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/scenes', icon: Layers, label: 'Scenes' },
  { to: '/preprocessing', icon: Wrench, label: 'Preprocessing' },
  { to: '/models', icon: GitBranch, label: 'Models' },
  { to: '/training', icon: Brain, label: 'Training' },
  { to: '/evaluation', icon: BarChart3, label: 'Evaluation' },
  { to: '/outputs', icon: Image, label: 'Outputs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export function Sidebar() {
  const location = useLocation()
  const storagePct = (dashboardStats.storageUsedBytes / dashboardStats.storageTotalBytes) * 100

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-[68px] flex-col border-r border-border bg-surface">
      <div className="flex h-14 items-center justify-center border-b border-border">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10">
          <Orbit className="h-4 w-4 text-primary" />
        </div>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto p-2">
        {navItems.map(({ to, icon: Icon, label, end }) => {
          const active = end ? location.pathname === to : location.pathname.startsWith(to)
          return (
            <NavLink key={to} to={to} end={end} title={label} className="relative block">
              {active && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute inset-0 rounded-xl bg-primary/10"
                  transition={{ duration: 0.2 }}
                />
              )}
              <div
                className={cn(
                  'relative flex h-10 w-full items-center justify-center rounded-xl transition-colors',
                  active ? 'text-primary' : 'text-muted hover:text-foreground hover:bg-card'
                )}
              >
                <Icon className="h-[18px] w-[18px]" strokeWidth={1.75} />
              </div>
            </NavLink>
          )
        })}
      </nav>

      <div className="border-t border-border p-2 space-y-2">
        <div className="rounded-xl bg-card border border-border p-2.5 space-y-2" title={`Storage: ${formatBytes(dashboardStats.storageUsedBytes)}`}>
          <div className="flex items-center gap-2">
            <HardDrive className="h-3 w-3 text-muted shrink-0" />
            <div className="flex-1 h-1 rounded-full bg-border overflow-hidden">
              <div className="h-full bg-primary rounded-full" style={{ width: `${storagePct}%` }} />
            </div>
          </div>
        </div>
        <div
          className="flex h-10 items-center justify-center rounded-xl"
          title={`${dashboardStats.gpuName} — ${dashboardStats.gpuAvailable ? 'Online' : 'Offline'}`}
        >
          <Cpu className={cn('h-[18px] w-[18px]', dashboardStats.gpuAvailable ? 'text-success' : 'text-error')} strokeWidth={1.75} />
        </div>
        <div className="text-center pb-1">
          <span className="text-[9px] text-muted-dim font-mono">v2.1</span>
        </div>
      </div>
    </aside>
  )
}
