import { useQuery } from '@tanstack/react-query'
import { getLoyaltySummary } from '../api/loyalty'

export function useLoyaltySummary() {
  return useQuery({ queryKey: ['loyaltySummary'], queryFn: getLoyaltySummary })
}
