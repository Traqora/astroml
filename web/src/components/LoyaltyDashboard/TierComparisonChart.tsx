import { memo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getTierComparison } from '../../api/loyalty'
import { VirtualizedTooltip } from '../charts/VirtualizedTooltip'
import { createChartConfig } from '../../lib/chartUtils'

const chartConfig = createChartConfig()

// Tier comparison is inherently a small dataset (a handful of tiers),
// so no downsampling is needed — just add memo + animation off.
export const TierComparisonChart = memo(function TierComparisonChart() {
  const { t } = useTranslation()
  const { data } = useQuery({ queryKey: ['tierComparison'], queryFn: getTierComparison })

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>{t('loyalty.tiers.comparison')}</h2>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={data ?? []} {...chartConfig}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tier" />
            <YAxis />
            <Tooltip content={<VirtualizedTooltip />} />
            <Legend />
            <Bar dataKey="threshold" name={t('loyalty.tiers.threshold')} fill="#8884d8" isAnimationActive={false} />
            <Bar dataKey="multiplier" name={t('loyalty.tiers.multiplier')} fill="#82ca9d" isAnimationActive={false} />
            <Bar dataKey="retention" name={t('loyalty.tiers.retention')} fill="#ffc658" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
})