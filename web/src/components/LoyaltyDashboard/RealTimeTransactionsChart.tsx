import { memo, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useIncomingTransactions } from '../../hooks/useIncomingTransactions'
import { VirtualizedTooltip } from '../charts/VirtualizedTooltip'
import { createChartConfig, sampleData, CHART_TARGET_POINTS } from '../../lib/chartUtils'

type ChartPoint = {
  id: string
  time: string
  amount: number
}

const chartConfig = createChartConfig()

const tooltipFormatter = (value: number) => `${value.toFixed(2)} XLM`

export const RealTimeTransactionsChart = memo(function RealTimeTransactionsChart() {
  const { t } = useTranslation()
  const transactions = useIncomingTransactions()

  // Memoize the chart data transformation + downsampling so it only recomputes
  // when `transactions` reference changes (i.e. each new WebSocket message).
  const chartData = useMemo<ChartPoint[]>(() => {
    const points: ChartPoint[] = [...transactions].reverse().map((tx) => ({
      id: tx.id,
      time: new Date(tx.timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }),
      amount: tx.amount,
    }))

    // Downsample only when we accumulate a large number of points
    return sampleData(points, CHART_TARGET_POINTS, 'amount')
  }, [transactions])

  const latest = transactions[0]

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          gap: 12,
          flexWrap: 'wrap',
        }}
      >
        <h2 style={{ margin: '8px 0' }}>{t('realtime.title')}</h2>
        <div style={{ color: 'var(--text-secondary, #555)', fontSize: 14 }}>
          {latest
            ? t('realtime.latest', { amount: latest.amount.toFixed(2), source: latest.sourceAccount })
            : t('realtime.waiting')}
        </div>
      </div>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} {...chartConfig}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" minTickGap={20} />
            <YAxis />
            <Tooltip
              content={<VirtualizedTooltip formatter={tooltipFormatter} />}
            />
            <Line
              type="monotone"
              dataKey="amount"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
})