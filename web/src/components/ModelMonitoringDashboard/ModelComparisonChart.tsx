import { memo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { VirtualizedTooltip } from '../charts/VirtualizedTooltip'
import { createChartConfig } from '../../lib/chartUtils'

interface ModelRun {
  id: string
  name: string
  color: string
  data: Array<{ date: string; accuracy: number }>
}

interface ModelComparisonChartProps {
  runs: ModelRun[]
  height?: number
}

const chartConfig = createChartConfig()
const accuracyFormatter = (value: number) => `${(value * 100).toFixed(1)}%`

const COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#eab308', '#8b5cf6']

export const ModelComparisonChart = memo(function ModelComparisonChart({
  runs,
  height = 240,
}: ModelComparisonChartProps) {
  const { t } = useTranslation()

  if (!runs || runs.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#888' }}>
        {t('monitoring.comparison.no_data')}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart {...chartConfig}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis domain={[0.7, 1.0]} tickFormatter={accuracyFormatter} />
        <Tooltip content={<VirtualizedTooltip formatter={accuracyFormatter} />} />
        <Legend />
        {runs.map((run, index) => (
          <Line
            key={run.id}
            type="monotone"
            dataKey="accuracy"
            data={run.data}
            name={run.name}
            stroke={run.color || COLORS[index % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
})