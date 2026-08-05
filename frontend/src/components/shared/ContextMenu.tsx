import { useEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'

interface ContextMenuProps {
  items: { label: string; icon?: React.ReactNode; danger?: boolean; onClick: () => void }[]
  children: React.ReactNode
}

export function ContextMenu({ items, children }: ContextMenuProps) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const close = () => setOpen(false)
    if (open) {
      document.addEventListener('click', close)
      return () => document.removeEventListener('click', close)
    }
  }, [open])

  return (
    <div
      ref={ref}
      onContextMenu={(e) => {
        e.preventDefault()
        setPos({ x: e.clientX, y: e.clientY })
        setOpen(true)
      }}
    >
      {children}
      {open && (
        <div
          className="fixed z-50 min-w-[160px] rounded-xl border border-border bg-card shadow-elevated py-1 animate-in fade-in duration-100"
          style={{ left: pos.x, top: pos.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {items.map((item) => (
            <button
              key={item.label}
              onClick={() => { item.onClick(); setOpen(false) }}
              className={cn(
                'flex w-full items-center gap-2 px-3 py-2 text-sm transition-colors hover:bg-surface',
                item.danger ? 'text-error' : 'text-foreground'
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
