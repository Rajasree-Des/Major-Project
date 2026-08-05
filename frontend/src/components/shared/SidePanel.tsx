import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface SidePanelProps {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  className?: string
}

export function SidePanel({ open, onClose, title, children, className }: SidePanelProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/40"
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className={cn(
              'fixed right-0 top-0 z-50 h-full w-full max-w-md border-l border-border bg-card shadow-elevated overflow-y-auto',
              className
            )}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card/95 backdrop-blur-xl px-6 py-4">
              <h2 className="text-sm font-semibold">{title}</h2>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-6">{children}</div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
