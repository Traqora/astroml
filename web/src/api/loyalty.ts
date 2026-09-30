import type {
  LoyaltySummary,
  LoyaltyTier,
  PointsTransaction,
  PointsHistoryResponse,
  RedemptionRequest,
  RedemptionResponse,
  StellarTransaction,
  TierComparisonDatum,
  FraudStats,
} from '../lib/types'
import { get, post, getAuthToken } from './client'
import { ApiError } from './client'

// Account ID for the current user (in a real app, this would come from auth)
const ACCOUNT_ID = import.meta.env.VITE_ACCOUNT_ID || 'GABC1234567890DEF'

/**
 * Get loyalty summary for the current account
 */
export async function getLoyaltySummary(): Promise<LoyaltySummary> {
  try {
    const response = await get<any>(`/api/v1/loyalty/${ACCOUNT_ID}`)
    
    // Transform API response to frontend format
    const currentTier: LoyaltyTier = {
      id: response.current_tier.id,
      name: response.current_tier.name,
      threshold: response.current_tier.threshold,
      multiplier: response.current_tier.multiplier,
      color: response.current_tier.color,
    }

    const nextTier = response.next_tier ? {
      tier: {
        id: response.next_tier.tier.id,
        name: response.next_tier.tier.name,
        threshold: response.next_tier.tier.threshold,
        multiplier: response.next_tier.tier.multiplier,
        color: response.next_tier.tier.color,
      },
      remainingToUpgrade: response.next_tier.remaining_to_upgrade,
      progressPct: response.next_tier.progress_pct,
    } : undefined

    const benefits = response.benefits.map((b: any) => ({
      id: b.id,
      title: b.title,
      description: b.description,
    }))

    return {
      currentTier,
      pointsBalance: response.points_balance,
      nextTier,
      benefits,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      // Return default values if loyalty data not found
      return {
        currentTier: { id: 'bronze', name: 'Bronze', threshold: 0, multiplier: 1.0, color: '#cd7f32' },
        pointsBalance: 0,
        benefits: [],
      }
    }
    throw error
  }
}

/**
 * Get points history for the current account
 */
export async function getPointsHistory(page: number, pageSize: number): Promise<PointsHistoryResponse> {
  const response = await get<any>(`/api/v1/loyalty/${ACCOUNT_ID}/history?page=${page}&page_size=${pageSize}`)
  
  const data = response.data.map((tx: any) => ({
    id: tx.id,
    date: tx.created_at,
    type: tx.type,
    points: tx.points,
    source: tx.source,
    note: tx.note,
  }))

  return {
    data,
    page: response.page,
    pageSize: response.page_size,
    total: response.total,
  }
}

/**
 * Redeem points for a reward
 */
export async function redeemPoints(req: RedemptionRequest): Promise<RedemptionResponse> {
  const response = await post<any>(`/api/v1/loyalty/${ACCOUNT_ID}/redeem`, req)
  
  return {
    newBalance: response.new_balance,
    transaction: {
      id: response.transaction.id,
      date: response.transaction.created_at,
      type: response.transaction.type,
      points: response.transaction.points,
      source: response.transaction.source,
      note: response.transaction.note,
    },
  }
}

/**
 * Get tier comparison data
 */
export async function getTierComparison(): Promise<TierComparisonDatum[]> {
  const response = await get<any>('/api/v1/loyalty/tiers')
  
  return response.map((tier: any) => ({
    tier: tier.name,
    threshold: tier.threshold,
    multiplier: tier.multiplier,
    retention: tier.retention || 0,
  }))
}

/**
 * Get referral link for the current account
 */
export async function getReferralLink(): Promise<{ url: string; invited: number; rewards: number }> {
  const response = await get<any>(`/api/v1/loyalty/${ACCOUNT_ID}/referral`)
  
  return {
    url: response.url,
    invited: response.invited,
    rewards: response.rewards,
  }
}

/**
 * Subscribe to incoming transactions via WebSocket
 * This is a placeholder for WebSocket integration
 */
type IncomingTransactionListener = (transaction: StellarTransaction) => void

function wsBaseUrl(): string {
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  return apiBase.replace(/^http/, 'ws') + '/api/v1/ws/transactions'
}

export function subscribeToIncomingTransactions(listener: IncomingTransactionListener): () => void {
  const token = getAuthToken()
  const url = token ? `${wsBaseUrl()}?token=${encodeURIComponent(token)}` : wsBaseUrl()
  let ws: WebSocket | null = null
  let closed = false

  try {
    ws = new WebSocket(url)
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'transaction' && msg.data) {
          listener(msg.data as StellarTransaction)
        } else if (msg.type === 'ping') {
          ws?.send('pong')
        }
      } catch {
        // ignore malformed messages
      }
    }
  } catch {
    // WebSocket unavailable — no-op cleanup
  }

  return () => {
    closed = true
    ws?.close()
    ws = null
    void closed
  }
}

/**
 * Get fraud statistics
 */
export async function getFraudStats(): Promise<FraudStats> {
  const response = await get<any>('/api/v1/fraud/stats')
  
  const recentAlerts = response.recent_alerts.map((alert: any) => ({
    id: alert.id,
    accountId: alert.account_id,
    pattern: alert.pattern,
    riskScore: alert.risk_score,
    detectedAt: alert.detected_at,
    description: alert.description,
  }))

  const riskOverTime = response.risk_over_time.map((point: any) => ({
    date: point.date,
    score: point.score,
  }))

  return {
    totalAlerts: response.total_alerts,
    highRisk: response.high_risk,
    mediumRisk: response.medium_risk,
    lowRisk: response.low_risk,
    recentAlerts,
    riskOverTime,
  }
}
