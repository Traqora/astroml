import { useQuery } from '@tanstack/react-query'
import { getTransactionByHash, getTransactionStats } from '../api/transactions'

export function useTransactionDetail(hash: string) {
  return useQuery({
    queryKey: ['transaction', hash],
    queryFn: () => getTransactionByHash(hash),
    enabled: !!hash,
  })
}

export function useTransactionStats() {
  return useQuery({
    queryKey: ['transactionStats'],
    queryFn: getTransactionStats,
  })
}
