import { FrozenDisclaimer } from '@/components/thermascope/FrozenDisclaimer'

export function AboutPage() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-text">Research & About</h1>
        <p className="mt-2 text-sm text-muted leading-relaxed">
          ThermaScope implements a physics-aware deep learning workflow for synthetic ×2 thermal super-resolution and
          RGB-guided colorization on Landsat QA-aware patches.
        </p>
      </div>

      <FrozenDisclaimer />

      <section className="rounded-xl border border-border-slate bg-surface p-6 space-y-3 text-sm text-muted leading-relaxed">
        <h2 className="text-base font-semibold text-text">Architecture</h2>
        <pre className="text-[10px] font-mono bg-midnight p-4 rounded-lg overflow-x-auto text-accent/90">{`
Landsat QA patches → synthetic LR (blur σ=1.2, noise 0.05 K)
        → SwinIR-S ×2 (pretrained, 1-ch adapted)
        → consistency refinement (β=1.0)
        → Kelvin + colorization A/B/C
        → metrics vs 30 m HR reference (eval only)`}</pre>
      </section>

      <section className="rounded-xl border border-border-slate bg-surface p-6 text-sm text-muted space-y-2">
        <h2 className="text-base font-semibold text-text">Verified locked-test summary (from frozen reports)</h2>
        <ul className="list-disc pl-5 space-y-1">
          <li>Multiscene SwinIR pooled RMSE ~0.31 K on locked Deccan+Assam test (see Results Lab).</li>
          <li>Physics refinement pooled RMSE ~0.24 K on same locked test with β=1.0.</li>
          <li>Kerala validation used for β selection only — not repeated tuning on locked scenes.</li>
        </ul>
      </section>

      <section className="rounded-xl border border-border-slate bg-surface p-6 text-sm text-muted space-y-2">
        <h2 className="text-base font-semibold text-text">Limitations</h2>
        <ul className="list-disc pl-5 space-y-1">
          <li>Not real 15 m thermal accuracy — 30 m reference and synthetic degradation.</li>
          <li>Refinement is forward-consistency under synthetic training, not radiative transfer.</li>
          <li>Pretrained SwinIR uses generic RGB SR weights adapted to thermal.</li>
          <li>RGB colorization is visualization-only.</li>
        </ul>
      </section>
    </div>
  )
}
