export interface StatusBadgeProps {
  text: string
  variant?: 'default' | 'active' | 'success' | 'warning'
}

export function StatusBadge({ text, variant = 'default' }: StatusBadgeProps) {
  const baseStyles =
    'inline-block px-2 py-1 rounded text-xs font-mono border'

  const variants = {
    default: 'border-muted bg-muted text-muted-foreground',
    active: 'border-primary bg-primary/20 text-primary',
    success: 'border-green-600 bg-green-950 text-green-300',
    warning: 'border-yellow-600 bg-yellow-950 text-yellow-300',
  }

  return <span className={`${baseStyles} ${variants[variant]}`}>{text}</span>
}
