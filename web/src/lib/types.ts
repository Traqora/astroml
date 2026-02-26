export type LoyaltyTier = {
  id: string
  name: string
  threshold: number
  multiplier: number
  color: string
}

export type Benefit = {
  id: string
  title: string
  description: string
  icon?: string
}

export type LoyaltySummary = {
  currentTier: LoyaltyTier
  pointsBalance: number
  nextTier?: {
    tier: LoyaltyTier
    remainingToUpgrade: number
    progressPct: number // 0..100
  }
  benefits: Benefit[]
}

export type PointsTransaction = {
  id: string
  date: string // ISO
  type: 'earn' | 'redeem' | 'adjust'
  points: number
  source?: string
  note?: string
}

export type PointsHistoryResponse = {
  data: PointsTransaction[]
  page: number
  pageSize: number
  total: number
}

export type RedemptionRequest = {
  points: number
  rewardId?: string
}

export type RedemptionResponse = {
  newBalance: number
  transaction: PointsTransaction
}

export type TierComparisonDatum = {
  tier: string
  threshold: number
  multiplier: number
  retention: number
}
