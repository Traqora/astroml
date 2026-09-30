import type { BlockchainTransaction, TransactionHistoryResponse } from '../lib/types'
import { get } from './client'
import { ApiError } from './client'

export interface AccountDetail {
  accountId: string
  balance: number | null
  sequence: number | null
  homeDomain: string | null
  flags: number
  lastModifiedLedger: number | null
  createdAt: string | null
  updatedAt: string | null
}

export interface AccountFraudSummary {
  accountId: string
  totalAlerts: number
  highRisk: number
  mediumRisk: number
  lowRisk: number
  latestScore: number | null
}

export interface AccountLoyaltySummary {
  accountId: string
  pointsBalance: number
  tierId: string
  tierName: string
}

export async function getAccountDetail(publicKey: string): Promise<AccountDetail | null> {
  try {
    const response = await get<any>(`/api/v1/accounts/${publicKey}`)
    return {
      accountId: response.account_id,
      balance: response.balance,
      sequence: response.sequence,
      homeDomain: response.home_domain,
      flags: response.flags,
      lastModifiedLedger: response.last_modified_ledger,
      createdAt: response.created_at,
      updatedAt: response.updated_at,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null
    }
    throw error
  }
}

export async function getAccountFraudSummary(publicKey: string): Promise<AccountFraudSummary | null> {
  try {
    const response = await get<any>(`/api/v1/accounts/${publicKey}/fraud-summary`)
    return {
      accountId: response.account_id,
      totalAlerts: response.total_alerts,
      highRisk: response.high_risk,
      mediumRisk: response.medium_risk,
      lowRisk: response.low_risk,
      latestScore: response.latest_score,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null
    }
    throw error
  }
}

export async function getAccountLoyaltySummary(publicKey: string): Promise<AccountLoyaltySummary | null> {
  try {
    const response = await get<any>(`/api/v1/accounts/${publicKey}/loyalty`)
    return {
      accountId: response.account_id,
      pointsBalance: response.points_balance,
      tierId: response.tier_id,
      tierName: response.tier_name,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null
    }
    throw error
  }
}

export async function getAccountTransactions(
  publicKey: string,
  page: number,
  pageSize: number
): Promise<TransactionHistoryResponse> {
  const response = await get<any>(
    `/api/v1/accounts/${publicKey}/transactions?page=${page + 1}&page_size=${pageSize}`
  )

  const data = response.data.map((tx: any) => ({
    hash: tx.hash,
    ledgerSequence: tx.ledger_sequence,
    sourceAccount: tx.source_account,
    destinationAccount: tx.destination_account,
    amount: tx.amount,
    assetCode: tx.asset_code,
    assetIssuer: tx.asset_issuer,
    operationType: tx.operation_type,
    createdAt: tx.created_at,
    fee: tx.fee,
    successful: tx.successful,
    memoType: tx.memo_type,
  }))

  return {
    data,
    page: response.page,
    pageSize: response.page_size,
    total: response.total,
  }
}
