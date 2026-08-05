import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { Sidebar } from '@/components/layout/Sidebar'
import { CommandBar } from '@/components/layout/CommandBar'
import { CommandPalette, useCommandPalette } from '@/components/shared/CommandPalette'
import { PageTransition } from '@/components/shared/PageTransition'

export function AppLayout() {
  const { open, setOpen, close } = useCommandPalette()
  const location = useLocation()

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div className="ml-[68px] flex min-h-screen flex-col">
        <CommandBar onOpenCommand={() => setOpen(true)} />
        <main className="flex-1 p-8 max-w-[1400px]">
          <AnimatePresence mode="wait">
            <PageTransition key={location.pathname}>
              <Outlet context={{ openImportModal: () => setOpen(false) }} />
            </PageTransition>
          </AnimatePresence>
        </main>
      </div>
      <CommandPalette open={open} onClose={close} />
    </div>
  )
}
