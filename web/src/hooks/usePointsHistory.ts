import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { getPointsHistory } from '../api/loyalty'
import type { PointsHistoryResponse } from '../lib/types'

export function usePointsHistory(page: number, pageSize: number) {
  return useQuery<PointsHistoryResponse>({
    queryKey: ['pointsHistory', page, pageSize],
    queryFn: () => getPointsHistory(page, pageSize),
    placeholderData: keepPreviousData,
  })
}
