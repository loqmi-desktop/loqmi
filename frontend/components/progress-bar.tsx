export interface ProgressBarProps {
  value: number
  max?: number
  label?: string
  showPercent?: boolean
}

export function ProgressBar({
  value,
  max = 100,
  label,
  showPercent = true,
}: ProgressBarProps) {
  const percentage = Math.round((value / max) * 100)

  return (
    <div className="w-full">
      {label && <p className="text-xs text-muted-foreground mb-1">{label}</p>}
      <div className="flex items-center gap-2">
        <div className="flex-1 bg-secondary rounded border border-border overflow-hidden h-2">
          <div
            className="bg-primary h-full transition-all duration-300"
            style={{ width: `${percentage}%` }}
          />
        </div>
        {showPercent && (
          <span className="text-xs text-muted-foreground w-8 text-right">
            {percentage}%
          </span>
        )}
      </div>
    </div>
  )
}
