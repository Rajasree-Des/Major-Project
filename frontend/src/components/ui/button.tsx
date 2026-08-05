import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-40',
  {
    variants: {
      variant: {
        default: 'bg-primary text-white hover:bg-primary-hover shadow-soft',
        secondary: 'bg-card text-foreground border border-border hover:bg-card-hover hover:border-border-subtle',
        outline: 'border border-border bg-transparent hover:bg-card text-muted hover:text-foreground',
        ghost: 'hover:bg-card text-muted hover:text-foreground',
        success: 'bg-success/10 text-success border border-success/20 hover:bg-success/15',
        danger: 'bg-error/10 text-error border border-error/20 hover:bg-error/15',
      },
      size: {
        default: 'h-9 px-4',
        sm: 'h-8 px-3 text-xs rounded-lg',
        lg: 'h-10 px-5',
        icon: 'h-9 w-9 rounded-lg',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size, className }))} {...props} />
}

export { buttonVariants }
