import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

type Props = {
  title: string
  description?: string
  children?: ReactNode
  className?: string
}

export function PageHeader({ title, description, children, className }: Props) {
  return (
    <div className={cn('mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between', className)}>
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-workspace-fg">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-sm text-workspace-muted-fg leading-relaxed">{description}</p>}
      </div>
      {children}
    </div>
  )
}
