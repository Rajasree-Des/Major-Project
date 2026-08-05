import { cn } from '@/lib/utils'

export function Input({ className, type, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type={type}
      className={cn(
        'flex h-9 w-full rounded-xl border border-border bg-surface px-3 text-sm text-foreground transition-colors placeholder:text-muted-dim focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary/50 disabled:opacity-40',
        className
      )}
      {...props}
    />
  )
}
