/**
 * Skeleton component for Transaction History Page
 * Replaces the loading text with proper skeleton placeholders
 */
import { memo } from 'react'
import { SkeletonCard } from './SkeletonCard'
import { SkeletonTable } from './SkeletonTable'

export const SkeletonTransactionHistory = memo(function SkeletonTransactionHistory() {
  return (
    <div style={{ display: 'grid', gap: 24 }}>
      {/* Header */}
      <div>
        <div
          style={{
            width: '40%',
            height: 28,
            borderRadius: 4,
            background: '#e0e0e0',
            marginBottom: 8,
          }}
        />
        <div
          style={{
            width: '60%',
            height: 16,
            borderRadius: 4,
            background: '#e8e8e8',
          }}
        />
      </div>

      {/* Filter section */}
      <SkeletonCard height={180} />

      {/* Transaction Table */}
      <SkeletonTable rows={8} columns={6} />

      {/* Footer count */}
      <div
        style={{
          width: '30%',
          height: 14,
          borderRadius: 4,
          background: '#e0e0e0',
          margin: '0 auto',
        }}
      />
    </div>
  )
})