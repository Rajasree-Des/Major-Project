import { cn } from '@/lib/utils'

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number
  indicatorClassName?: string
  size?: 'sm' | 'md'
}

export function Progress({ value, className, indicatorClassName, size = 'md', ...props }: ProgressProps) {
  const clamped = Math.min(100, Math.max(0, value))
  return (
    <div
      className={cn(
        'relative w-full overflow-hidden rounded-full bg-border',
        size === 'sm' ? 'h-1' : 'h-1.5',
        className
      )}
      {...props}
    >
      <div
        className={cn('h-full rounded-full bg-primary transition-all duration-700 ease-out', indicatorClassName)}
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}

export function CircularProgress({ value, size = 48, strokeWidth = 3 }: { value: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference

  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#27272a" strokeWidth={strokeWidth} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="#3b82f6"
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="transition-all duration-700 ease-out"
      />
    </svg>
  )
}
