import { useEffect, useRef } from 'react'
import { RadialBarChart, RadialBar, ResponsiveContainer, Tooltip } from 'recharts'
import { fireConfetti } from '../../lib/confetti'
import type { LoyaltyTier } from '../../lib/types'

export function TierProgress({ currentTier, nextTier }: { currentTier: LoyaltyTier; nextTier?: { tier: LoyaltyTier; remainingToUpgrade: number; progressPct: number } }) {
  const prevTierId = useRef<string | null>(null)

  useEffect(() => {
    if (prevTierId.current && prevTierId.current !== currentTier.id) {
      fireConfetti()
    }
    prevTierId.current = currentTier.id
  }, [currentTier.id])

  const progress = nextTier?.progressPct ?? 100

  const data = [
    { name: 'Progress', value: progress, fill: currentTier.color },
  ]

  return (
    <div style={{ width: 260, height: 160, display: 'flex', alignItems: 'center', gap: 16 }}>
      <div style={{ width: 160, height: 160 }}>
        <ResponsiveContainer>
          <RadialBarChart innerRadius="70%" outerRadius="100%" data={data} startAngle={90} endAngle={-270}>
            <RadialBar minAngle={15} dataKey="value" cornerRadius={6} />
            <Tooltip formatter={(v: any) => `${v}%`} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div style={{ fontSize: 14, color: '#555' }}>Progress to Next Tier</div>
        <div style={{ fontSize: 24, fontWeight: 700 }}>{progress}%</div>
        {nextTier && (
          <div style={{ fontSize: 12, color: '#777' }}>{nextTier.remainingToUpgrade.toLocaleString()} points to {nextTier.tier.name}</div>
        )}
        {!nextTier && <div style={{ fontSize: 12, color: '#777' }}>Top tier achieved</div>}
      </div>
    </div>
  )}
