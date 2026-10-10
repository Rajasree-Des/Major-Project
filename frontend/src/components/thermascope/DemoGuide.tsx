import { Link } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { useState } from 'react'

type Props = {
  demoScene?: string
  demoPatch?: string
  demoAvailable?: boolean
}

export function DemoGuide({ demoScene, demoPatch, demoAvailable }: Props) {
  const [open, setOpen] = useState(false)
  const studioLink =
    demoAvailable && demoScene && demoPatch
      ? `/studio?scene=${encodeURIComponent(demoScene)}&patch=${encodeURIComponent(demoPatch)}`
      : '/studio'

  return (
    <div className="rounded-xl border border-border-slate bg-surface/80">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-text"
      >
        Live demo workflow
        <ChevronDown className={`h-4 w-4 text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <ol className="list-decimal space-y-2 border-t border-border-slate px-6 py-4 text-xs text-muted leading-relaxed">
          <li>Open Command Center — verify API connected (top bar).</li>
          <li>
            <Link to={studioLink} className="text-accent hover:underline">
              Processing Studio
            </Link>{' '}
            — select Kerala validation scene and patch {demoAvailable ? demoPatch : '(choose a valid patch)'}.
          </li>
          <li>Run Pipeline — wait for real CPU inference to complete.</li>
          <li>Inspect stage tabs: LR input, bicubic, SwinIR, refined, colorization.</li>
          <li>
            <Link to="/colorization" className="text-accent hover:underline">
              Colorization Studio
            </Link>{' '}
            — switch methods A/B/C.
          </li>
          <li>
            <Link to="/results" className="text-accent hover:underline">
              Results Lab
            </Link>{' '}
            — locked Deccan/Assam benchmarks.
          </li>
          <li>
            <Link to="/about" className="text-accent hover:underline">
              Research & About
            </Link>{' '}
            — limitations and architecture.
          </li>
        </ol>
      )}
    </div>
  )
}
