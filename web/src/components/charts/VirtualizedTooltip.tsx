/**
 * VirtualizedTooltip – a lightweight Recharts custom tooltip that avoids
 * constructing expensive DOM nodes when the tooltip is not active.
 *
 * Drop-in replacement for Recharts' built-in <Tooltip /> content prop.
 * Pass it as: <Tooltip content={<VirtualizedTooltip formatter={…} />} />
 */

import { memo } from 'react'

export interface VirtualizedTooltipProps {
  /** Return value is rendered as the "value" string next to the data key. */
  formatter?: (value: number, name: string) => string
  // Recharts injects these automatically when used as a Tooltip content prop
  active?: boolean
  payload?: Array<{ name: string; value: number; color?: string }>
  label?: string | number
}

/**
 * Renders nothing at all when `active` is false or `payload` is empty.
 * When active, renders a minimal container — no Recharts internals, no
 * extra SVG layers.
 */
export const VirtualizedTooltip = memo(function VirtualizedTooltip({
  active,
  payload,
  label,
  formatter,
}: VirtualizedTooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div style={containerStyle}>
      {label !== undefined && <div style={labelStyle}>{label}</div>}
      {payload.map((entry) => (
        <div key={entry.name} style={rowStyle}>
          <span style={{ ...dotStyle, background: entry.color ?? '#8884d8' }} />
          <span style={nameStyle}>{entry.name}:</span>
          <span style={valueStyle}>
            {formatter ? formatter(entry.value, entry.name) : String(entry.value)}
          </span>
        </div>
      ))}
    </div>
  )
})

// ---------------------------------------------------------------------------
// Styles (inline to avoid CSS bundle dependencies)
// ---------------------------------------------------------------------------

const containerStyle: React.CSSProperties = {
  background: 'var(--bg-card, #fff)',
  border: '1px solid var(--border-color, #e2e8f0)',
  borderRadius: 6,
  padding: '8px 12px',
  fontSize: 12,
  boxShadow: 'var(--shadow-md, 0 2px 8px rgba(0,0,0,0.10))',
  pointerEvents: 'none',
  maxWidth: 220,
}

const labelStyle: React.CSSProperties = {
  fontWeight: 600,
  marginBottom: 4,
  color: 'var(--text-secondary, #4a5568)',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  marginTop: 2,
}

const dotStyle: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: '50%',
  flexShrink: 0,
}

const nameStyle: React.CSSProperties = {
  color: 'var(--text-muted, #718096)',
  flexShrink: 0,
}

const valueStyle: React.CSSProperties = {
  fontWeight: 600,
  color: 'var(--text-primary, #1a202c)',
}
