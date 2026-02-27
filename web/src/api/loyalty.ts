import type {
  LoyaltySummary,
  PointsHistoryResponse,
  RedemptionRequest,
  RedemptionResponse,
  TierComparisonDatum,
} from '../lib/types'

// For demo purposes, use in-memory mock data. Replace with real HTTP calls later.
let pointsBalance = 3250
let currentTier = { id: 'gold', name: 'Gold', threshold: 3000, multiplier: 1.25, color: '#d4af37' }
const silver = { id: 'silver', name: 'Silver', threshold: 1500, multiplier: 1.1, color: '#c0c0c0' }
const platinum = { id: 'platinum', name: 'Platinum', threshold: 6000, multiplier: 1.5, color: '#e5e4e2' }

const history = Array.from({ length: 137 }).map((_, i) => {
  const earn = Math.floor(Math.random() * 200) + 20
  const date = new Date(Date.now() - i * 86400000).toISOString()
  return {
    id: `txn_${i}`,
    date,
    type: 'earn' as const,
    points: earn,
    source: 'Purchase',
  }
})

export async function getLoyaltySummary(): Promise<LoyaltySummary> {
  const nextTier = pointsBalance >= platinum.threshold
    ? undefined
    : {
        tier: pointsBalance >= silver.threshold ? platinum : goldLike(platinum),
        remainingToUpgrade: Math.max(0, (pointsBalance >= silver.threshold ? platinum.threshold : silver.threshold) - pointsBalance),
        progressPct: Math.min(100, Math.round((pointsBalance / (pointsBalance >= silver.threshold ? platinum.threshold : silver.threshold)) * 100)),
      }

  const benefits = [
    { id: 'b1', title: 'Free Shipping', description: 'No shipping fees on all orders.' },
    { id: 'b2', title: 'Birthday Bonus', description: '500 bonus points on your birthday.' },
    { id: 'b3', title: 'Priority Support', description: 'Skip the line with priority support.' },
  ]

  return {
    currentTier,
    pointsBalance,
    nextTier,
    benefits,
  }
}

function goldLike(p: { id: string; name: string }) {
  return { id: 'gold', name: 'Gold' }
}

export async function getPointsHistory(page: number, pageSize: number): Promise<PointsHistoryResponse> {
  const start = page * pageSize
  const end = start + pageSize
  const data = history.slice(start, end)
  return { data, page, pageSize, total: history.length }
}

export async function redeemPoints(req: RedemptionRequest): Promise<RedemptionResponse> {
  await delay(300)
  if (req.points <= 0 || req.points > pointsBalance) {
    throw new Error('Invalid redemption amount')
  }
  pointsBalance -= req.points
  const transaction = {
    id: `txn_red_${Date.now()}`,
    date: new Date().toISOString(),
    type: 'redeem' as const,
    points: -Math.abs(req.points),
    source: 'Redemption',
  }
  history.unshift(transaction)
  return { newBalance: pointsBalance, transaction }
}

export async function getTierComparison(): Promise<TierComparisonDatum[]> {
  return [
    { tier: 'Silver', threshold: 1500, multiplier: 1.1, retention: 70 },
    { tier: 'Gold', threshold: 3000, multiplier: 1.25, retention: 80 },
    { tier: 'Platinum', threshold: 6000, multiplier: 1.5, retention: 90 },
  ]
}

export async function getReferralLink(): Promise<{ url: string; invited: number; rewards: number }> {
  return { url: 'https://example.com/ref?code=ABC123', invited: 12, rewards: 4 }
}

function delay(ms: number) {
  return new Promise((res) => setTimeout(res, ms))
}
