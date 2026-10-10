import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

type Selection = {
  sceneId: string
  patchId: string
  setSceneId: (id: string) => void
  setPatchId: (id: string) => void
  setSelection: (sceneId: string, patchId: string) => void
}

const SelectionContext = createContext<Selection | null>(null)

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [sceneId, setSceneId] = useState('LC08_L2SP_144053_20250225_02_T1')
  const [patchId, setPatchId] = useState('000096')

  const value = useMemo(
    () => ({
      sceneId,
      patchId,
      setSceneId,
      setPatchId,
      setSelection: (s: string, p: string) => {
        setSceneId(s)
        setPatchId(p)
      },
    }),
    [sceneId, patchId]
  )

  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>
}

export function useSelection() {
  const ctx = useContext(SelectionContext)
  if (!ctx) throw new Error('useSelection requires SelectionProvider')
  return ctx
}
