/**
 * Skeleton component for chart placeholders with shimmer animation
 * Used while chart data is loading
 */
import { memo } from 'react'

interface SkeletonChartProps {
  height?: number
  width?: string | number
}

export const SkeletonChart = memo(function SkeletonChart({
  height = 240,
  width = '100%',
}: SkeletonChartProps) {
  return (
    <div
      style={{
        width,
        height,
        borderRadius: 16,
        background: '#fff',
        boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)',
        border: '1px solid #ececec',
        padding: 20,
        position: 'relative',
        overflow: 'hidden',
      }}
      role="status"
      aria-label="Loading chart"
    >
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div
          style={{
            width: '50%',
            height: 20,
            borderRadius: 4,
            background: '#e0e0e0',
            marginBottom: 16,
          }}
        />
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'flex-end',
            gap: 8,
            paddingBottom: 20,
          }}
        >
          {[...Array(10)].map((_, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${20 + Math.random() * 60}%`,
                borderRadius: '4px 4px 0 0',
                background: '#e0e0e0',
              }}
            />
          ))}
        </div>
        <div
          style={{
            display: 'flex',
            gap: 16,
            marginTop: 12,
            paddingTop: 12,
            borderTop: '1px solid #f0f0f0',
          }}
        >
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: 12,
                borderRadius: 4,
                background: '#e0e0e0',
              }}
            />
          ))}
        </div>
      </div>
      <Shimmer />
    </div>
  )
})

/**
 * Shimmer animation component for skeleton loading effect
 */
function Shimmer() {
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background:
          'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
        animation: 'shimmer 1.5s infinite',
        transform: 'translateX(-100%)',
      }}
    />
  )
}