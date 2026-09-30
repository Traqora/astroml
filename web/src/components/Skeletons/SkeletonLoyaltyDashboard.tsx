/**
 * Skeleton component for Loyalty Dashboard
 * Replaces the loading text with proper skeleton placeholders
 */
import { memo } from 'react'
import { SkeletonCard } from './SkeletonCard'
import { SkeletonChart } from './SkeletonChart'
import { SkeletonTable } from './SkeletonTable'

export const SkeletonLoyaltyDashboard = memo(function SkeletonLoyaltyDashboard() {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {/* Tier summary section */}
      <section
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
          <SkeletonCard width={150} height={80} />
          <SkeletonCard width={150} height={80} />
        </div>
        <SkeletonCard width={200} height={80} />
      </section>

      {/* Benefits Card */}
      <section>
        <SkeletonCard height={150} />
      </section>

      {/* Redemption Panel */}
      <section>
        <SkeletonCard height={120} />
      </section>

      {/* Comparison Chart */}
      <section>
        <SkeletonChart height={300} />
      </section>

      {/* Real-time Chart */}
      <section>
        <SkeletonChart height={300} />
      </section>

      {/* Fraud Detection Panel */}
      <section>
        <SkeletonCard height={200} />
      </section>

      {/* Points History Table */}
      <section>
        <SkeletonTable rows={5} columns={5} />
      </section>

      {/* Referral Section */}
      <section>
        <SkeletonCard height={120} />
      </section>
    </div>
  )
})