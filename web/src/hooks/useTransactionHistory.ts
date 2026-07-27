import { useQuery } from '@tanstack/react-query'
import { getTransactionHistory } from '../api/transactions'
import type { TransactionHistoryResponse } from '../lib/types'

export function useTransactionHistory(
  page: number,
  pageSize: number,
  filters?: {
    sourceAccount?: string
    operationType?: string
    startDate?: string
    endDate?: string
  }
) {
  return useQuery({
    queryKey: ['transactions', page, pageSize, filters],
    queryFn: () => getTransactionHistory(page, pageSize, filters),
  })
}
