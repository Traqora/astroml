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

export type StellarTransaction = {
  id: string
  timestamp: string // ISO
  amount: number
  sourceAccount: string
  destinationAccount: string
}

export type FraudAlert = {
  id: string
  accountId: string
  pattern: 'sybil_cluster' | 'wash_trading_loop' | 'anomaly'
  riskScore: number // 0..100
  detectedAt: string // ISO
  description: string
}

export type FraudStats = {
  totalAlerts: number
  highRisk: number
  mediumRisk: number
  lowRisk: number
  recentAlerts: FraudAlert[]
  riskOverTime: { date: string; score: number }[]
}

export type BlockchainTransaction = {
  hash: string
  ledgerSequence: number
  sourceAccount: string
  destinationAccount?: string
  amount?: number
  assetCode?: string
  assetIssuer?: string
  operationType: string
  createdAt: string // ISO
  fee: number
  successful: boolean
  memoType?: string
}

export type TransactionHistoryResponse = {
  data: BlockchainTransaction[]
  page: number
  pageSize: number
  total: number
}
