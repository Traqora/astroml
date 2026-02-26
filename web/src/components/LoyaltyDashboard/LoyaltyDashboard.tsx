import { useState, useRef, useEffect } from 'react'
import { useLoyaltySummary } from '../../hooks/useLoyaltySummary'
import { usePointsHistory } from '../../hooks/usePointsHistory'
import { useRedeemPoints } from '../../hooks/useRedeemPoints'
import { TierProgress } from './TierProgress'
import { PointsHistoryTable } from './PointsHistoryTable'
import { TierBenefitsCard } from './TierBenefitsCard'
import { PointsRedemptionPanel } from './PointsRedemptionPanel'
import { TierComparisonChart } from './TierComparisonChart'
import { ReferralInviteSection } from './ReferralInviteSection'

export function LoyaltyDashboard() {
  const { data: summary, isLoading: loadingSummary } = useLoyaltySummary()

  const [page, setPage] = useState(0)
  const pageSize = 10
  const { data: history, isLoading: loadingHistory } = usePointsHistory(page, pageSize)

  const redeem = useRedeemPoints(page, pageSize)

  const prevTierId = useRef<string | null>(null)
  useEffect(() => {
    if (summary?.currentTier?.id && prevTierId.current && prevTierId.current !== summary.currentTier.id) {
      // Tier upgrade handled inside TierProgress via prop change; this is just to keep prev value
    }
    if (summary?.currentTier?.id) {
      prevTierId.current = summary.currentTier.id
    }
  }, [summary?.currentTier?.id])

  if (loadingSummary || !summary) return <div>Loading...</div>

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <section style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 18, color: '#555' }}>Current Tier</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: summary.currentTier.color }}>{summary.currentTier.name}</div>
        </div>
        <div>
          <div style={{ fontSize: 18, color: '#555' }}>Points Balance</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{summary.pointsBalance.toLocaleString()}</div>
        </div>
        <TierProgress currentTier={summary.currentTier} nextTier={summary.nextTier} />
      </section>

      <section>
        <TierBenefitsCard benefits={summary.benefits} />
      </section>

      <section>
        <PointsRedemptionPanel
          balance={summary.pointsBalance}
          onRedeem={(points) => redeem.mutate({ points })}
          pending={redeem.isPending}
        />
      </section>

      <section>
        <TierComparisonChart />
      </section>

      <section>
        <PointsHistoryTable
          response={history}
          loading={loadingHistory}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
        />
      </section>

      <section>
        <ReferralInviteSection />
      </section>
    </div>
  )
}
