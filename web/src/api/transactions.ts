import type { BlockchainTransaction, TransactionHistoryResponse } from '../lib/types'
import { get } from './client'
import { ApiError } from './client'

/**
 * Get transaction history with optional filters
 */
export async function getTransactionHistory(
  page: number,
  pageSize: number,
  filters?: {
    sourceAccount?: string
    destinationAccount?: string
    assetCode?: string
    startDate?: string
    endDate?: string
    minAmount?: number
    maxAmount?: number
    operationType?: string
    successful?: boolean
  }
): Promise<TransactionHistoryResponse> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  })

  if (filters?.sourceAccount) {
    params.append('source_account', filters.sourceAccount)
  }
  if (filters?.destinationAccount) {
    params.append('destination_account', filters.destinationAccount)
  }
  if (filters?.assetCode) {
    params.append('asset_code', filters.assetCode)
  }
  if (filters?.startDate) {
    params.append('start_date', filters.startDate)
  }
  if (filters?.endDate) {
    params.append('end_date', filters.endDate)
  }
  if (filters?.minAmount !== undefined) {
    params.append('min_amount', filters.minAmount.toString())
  }
  if (filters?.maxAmount !== undefined) {
    params.append('max_amount', filters.maxAmount.toString())
  }
  if (filters?.operationType) {
    params.append('operation_type', filters.operationType)
  }
  if (filters?.successful !== undefined) {
    params.append('successful', filters.successful.toString())
  }

  const response = await get<any>(`/api/v1/transactions?${params.toString()}`)

  const data = response.data.map((tx: any) => ({
    hash: tx.hash,
    ledgerSequence: tx.ledgerSequence,
    sourceAccount: tx.sourceAccount,
    destinationAccount: tx.destinationAccount,
    amount: tx.amount,
    assetCode: tx.assetCode,
    assetIssuer: tx.assetIssuer,
    operationType: tx.operationType,
    createdAt: tx.createdAt,
    fee: tx.fee,
    successful: tx.successful,
    memoType: tx.memoType,
  }))

  return {
    data,
    page: response.page,
    pageSize: response.pageSize,
    total: response.total,
  }
}

/**
 * Get a single transaction by hash
 */
export async function getTransactionByHash(hash: string): Promise<BlockchainTransaction | null> {
  try {
    const response = await get<any>(`/api/v1/transactions/${hash}`)

    return {
      hash: response.hash,
      ledgerSequence: response.ledgerSequence,
      sourceAccount: response.sourceAccount,
      destinationAccount: response.destinationAccount,
      amount: response.amount,
      assetCode: response.assetCode,
      assetIssuer: response.assetIssuer,
      operationType: response.operationType,
      createdAt: response.createdAt,
      fee: response.fee,
      successful: response.successful,
      memoType: response.memoType,
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null
    }
    throw error
  }
}

/**
 * Get transaction statistics
 */
export async function getTransactionStats(): Promise<{
  totalCount: number
  totalVolume: number
  countByAsset: Record<string, number>
  successfulCount: number
  failedCount: number
}> {
  const response = await get<any>('/api/v1/transactions/stats')

  return {
    totalCount: response.total_count,
    totalVolume: response.total_volume,
    countByAsset: response.count_by_asset,
    successfulCount: response.successful_count,
    failedCount: response.failed_count,
  }
}
