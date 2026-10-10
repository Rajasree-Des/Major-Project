import { useEffect, useState } from 'react'
import { api, type Health } from '@/lib/api'

export function useHealth(pollMs = 5000) {
  const [health, setHealth] = useState<Health | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const h = await api.health()
        if (!cancelled) {
          setHealth(h)
          setConnected(true)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) {
          setConnected(false)
          setError(e instanceof Error ? e.message : 'Backend unreachable')
        }
      }
    }

    poll()
    const t = setInterval(poll, pollMs)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [pollMs])

  return { health, connected, error }
}
