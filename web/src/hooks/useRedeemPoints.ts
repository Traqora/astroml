import { useMutation, useQueryClient } from '@tanstack/react-query'
import { redeemPoints } from '../api/loyalty'
import type { RedemptionRequest } from '../lib/types'

export function useRedeemPoints(currentPage: number, pageSize: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: RedemptionRequest) => redeemPoints(req),
    onMutate: async (variables) => {
      await Promise.all([
        qc.cancelQueries({ queryKey: ['loyaltySummary'] }),
        qc.cancelQueries({ queryKey: ['pointsHistory', currentPage, pageSize] }),
      ])

      const prevSummary = qc.getQueryData<any>(['loyaltySummary'])
      const prevHistory = qc.getQueryData<any>(['pointsHistory', currentPage, pageSize])

      // Optimistic updates
      if (prevSummary) {
        qc.setQueryData(['loyaltySummary'], {
          ...prevSummary,
          pointsBalance: prevSummary.pointsBalance - variables.points,
        })
      }

      if (prevHistory) {
        const tempTxn = {
          id: 'temp-redemption',
          date: new Date().toISOString(),
          type: 'redeem' as const,
          points: -Math.abs(variables.points),
          source: 'Redemption (pending)',
        }
        qc.setQueryData(['pointsHistory', currentPage, pageSize], {
          ...prevHistory,
          data: [tempTxn, ...prevHistory.data],
          total: prevHistory.total + 1,
        })
      }

      return { prevSummary, prevHistory }
    },
    onError: (err, _variables, ctx) => {
      if (!ctx) return
      qc.setQueryData(['loyaltySummary'], ctx.prevSummary)
      qc.setQueryData(['pointsHistory', currentPage, pageSize], ctx.prevHistory)
    },
    onSuccess: (data) => {
      // Reconcile with server values
      qc.setQueryData(['loyaltySummary'], (prev: any) => ({ ...prev, pointsBalance: data.newBalance }))
      qc.setQueryData(['pointsHistory', currentPage, pageSize], (prev: any) => {
        const withoutTemp = prev.data.filter((d: any) => d.id !== 'temp-redemption')
        return { ...prev, data: [data.transaction, ...withoutTemp] }
      })
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['loyaltySummary'] })
      qc.invalidateQueries({ queryKey: ['pointsHistory', currentPage, pageSize] })
    },
  })
}
