import { useQuery } from '@tanstack/react-query'
import { getPointsHistory } from '../api/loyalty'

export function usePointsHistory(page: number, pageSize: number) {
  return useQuery({
    queryKey: ['pointsHistory', page, pageSize],
    queryFn: () => getPointsHistory(page, pageSize),
    keepPreviousData: true,
  })
}
