/**
 * Skeleton component for metric cards with shimmer animation
 * Used for displaying placeholder while content loads
 */
import { memo } from 'react'

interface SkeletonCardProps {
  width?: string | number
  height?: string | number
  borderRadius?: string | number
}

export const SkeletonCard = memo(function SkeletonCard({
  width = '100%',
  height = 120,
  borderRadius = 16,
}: SkeletonCardProps) {
  return (
    <div
      style={{
        width,
        height,
        borderRadius,
        background: '#fff',
        boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)',
        border: '1px solid #ececec',
        padding: 20,
        position: 'relative',
        overflow: 'hidden',
      }}
      role="status"
      aria-label="Loading content"
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          height: '100%',
        }}
      >
        <div
          style={{
            width: '60%',
            height: 14,
            borderRadius: 4,
            background: '#e0e0e0',
          }}
        />
        <div
          style={{
            width: '40%',
            height: 28,
            borderRadius: 4,
            background: '#e0e0e0',
            marginTop: 4,
          }}
        />
        <div
          style={{
            width: '80%',
            height: 12,
            borderRadius: 4,
            background: '#e0e0e0',
            marginTop: 'auto',
          }}
        />
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