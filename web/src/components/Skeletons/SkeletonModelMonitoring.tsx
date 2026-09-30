/**
 * Skeleton component for Model Monitoring Dashboard
 * Replaces the loading text with proper skeleton placeholders
 */
import { memo } from 'react'
import { SkeletonCard } from './SkeletonCard'
import { SkeletonChart } from './SkeletonChart'

export const SkeletonModelMonitoring = memo(function SkeletonModelMonitoring() {
  return (
    <section style={{ display: 'grid', gap: 24 }}>
      {/* Metric cards grid - 4 cards */}
      <div
        style={{
          display: 'grid',
          gap: 16,
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        }}
      >
        {[...Array(4)].map((_, i) => (
          <SkeletonCard key={i} height={120} />
        ))}
      </div>

      {/* Charts grid - 2 charts */}
      <div
        style={{
          display: 'grid',
          gap: 24,
          gridTemplateColumns: '1.5fr 1fr',
        }}
      >
        <SkeletonChart height={320} />
        <SkeletonChart height={320} />
      </div>
    </section>
  )
})