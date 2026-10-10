import { Outlet } from 'react-router-dom'
import { Sidebar } from '@/components/layout/Sidebar'
import { TopBar } from '@/components/layout/TopBar'
import { SelectionProvider } from '@/context/SelectionContext'
import { useEffect, useState } from 'react'

export function AppLayout() {
  const [ml, setMl] = useState('248px')

  useEffect(() => {
    const read = () => setMl(localStorage.getItem('ts-sidebar') === 'collapsed' ? '72px' : '248px')
    read()
    window.addEventListener('storage', read)
    const t = setInterval(read, 300)
    return () => {
      window.removeEventListener('storage', read)
      clearInterval(t)
    }
  }, [])

  return (
    <SelectionProvider>
      <div className="min-h-screen bg-midnight grid-scientific">
        <Sidebar />
        <div className="flex min-h-screen flex-col transition-[margin] duration-200" style={{ marginLeft: ml }}>
          <TopBar />
          <main className="flex-1 p-6 lg:p-8 max-w-[1400px] w-full">
            <Outlet />
          </main>
        </div>
      </div>
    </SelectionProvider>
  )
}
