import { useQuery } from '@tanstack/react-query'
import { getAccountDetail, getAccountFraudSummary, getAccountLoyaltySummary, getAccountTransactions } from '../api/accounts'

export function useAccountDetail(publicKey: string) {
  return useQuery({
    queryKey: ['account', publicKey],
    queryFn: () => getAccountDetail(publicKey),
    enabled: !!publicKey,
  })
}

export function useAccountFraudSummary(publicKey: string) {
  return useQuery({
    queryKey: ['accountFraud', publicKey],
    queryFn: () => getAccountFraudSummary(publicKey),
    enabled: !!publicKey,
  })
}

export function useAccountLoyaltySummary(publicKey: string) {
  return useQuery({
    queryKey: ['accountLoyalty', publicKey],
    queryFn: () => getAccountLoyaltySummary(publicKey),
    enabled: !!publicKey,
  })
}

export function useAccountTransactions(publicKey: string, page: number, pageSize: number) {
  return useQuery({
    queryKey: ['accountTransactions', publicKey, page, pageSize],
    queryFn: () => getAccountTransactions(publicKey, page, pageSize),
    enabled: !!publicKey,
  })
}
