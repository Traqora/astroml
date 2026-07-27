/**
 * Skeleton component for table placeholders with shimmer animation
 * Used while table data is loading
 */
import { memo } from 'react'

interface SkeletonTableProps {
  rows?: number
  columns?: number
}

export const SkeletonTable = memo(function SkeletonTable({
  rows = 5,
  columns = 4,
}: SkeletonTableProps) {
  return (
    <div
      style={{
        borderRadius: 16,
        background: '#fff',
        boxShadow: '0 2px 14px rgba(0, 0, 0, 0.06)',
        border: '1px solid #ececec',
        padding: 20,
        position: 'relative',
        overflow: 'hidden',
      }}
      role="status"
      aria-label="Loading table"
    >
      {/* Table header */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${columns}, 1fr)`,
          gap: 16,
          paddingBottom: 12,
          borderBottom: '1px solid #f0f0f0',
          marginBottom: 12,
        }}
      >
        {[...Array(columns)].map((_, i) => (
          <div
            key={i}
            style={{
              height: 16,
              borderRadius: 4,
              background: '#d0d0d0',
              width: i === 0 ? '60%' : '80%',
            }}
          />
        ))}
      </div>

      {/* Table rows */}
      {[...Array(rows)].map((_, rowIndex) => (
        <div
          key={rowIndex}
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${columns}, 1fr)`,
            gap: 16,
            padding: '8px 0',
            borderBottom: rowIndex < rows - 1 ? '1px solid #f5f5f5' : 'none',
          }}
        >
          {[...Array(columns)].map((_, colIndex) => (
            <div
              key={colIndex}
              style={{
                height: 14,
                borderRadius: 4,
                background: '#e8e8e8',
                width: colIndex === 0 ? '70%' : colIndex === columns - 1 ? '50%' : '90%',
              }}
            />
          ))}
        </div>
      ))}

      {/* Pagination skeleton */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: 16,
          paddingTop: 12,
          borderTop: '1px solid #f0f0f0',
        }}
      >
        <div
          style={{
            width: '30%',
            height: 12,
            borderRadius: 4,
            background: '#e0e0e0',
          }}
        />
        <div
          style={{
            display: 'flex',
            gap: 8,
          }}
        >
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              style={{
                width: 32,
                height: 32,
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