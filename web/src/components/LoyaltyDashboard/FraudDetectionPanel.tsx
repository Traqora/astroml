import { memo, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
  Tooltip,
} from 'recharts'
import { getFraudStats } from '../../api/loyalty'
import type { FraudAlert } from '../../lib/types'
import { VirtualizedTooltip } from '../charts/VirtualizedTooltip'
import { createChartConfig, sampleData, CHART_TARGET_POINTS } from '../../lib/chartUtils'

const PATTERN_LABELS: Record<FraudAlert['pattern'], string> = {
  sybil_cluster: 'fraud.alerts.patterns.sybil_cluster',
  wash_trading_loop: 'fraud.alerts.patterns.wash_trading_loop',
  anomaly: 'fraud.alerts.patterns.anomaly',
}

const RISK_COLOR = (score: number) =>
  score >= 75 ? '#e53e3e' : score >= 50 ? '#dd6b20' : '#38a169'

const PIE_COLORS = ['#e53e3e', '#dd6b20', '#38a169']

const chartConfig = createChartConfig()
const riskTooltipFormatter = (value: number) => `${value.toFixed(1)}`

export const FraudDetectionPanel = memo(function FraudDetectionPanel() {
  const { t } = useTranslation()
  const { data, isLoading } = useQuery({ queryKey: ['fraudStats'], queryFn: getFraudStats })

  // Downsample risk-over-time series only when it exceeds the threshold
  const riskOverTime = useMemo(
    () =>
      data
        ? sampleData(data.riskOverTime, CHART_TARGET_POINTS, 'score')
        : [],
    [data]
  )

  const pieData = useMemo(
    () =>
      data
        ? [
            { name: t('fraud.stats.high_risk'), value: data.highRisk },
            { name: t('fraud.stats.medium_risk'), value: data.mediumRisk },
            { name: t('fraud.stats.low_risk'), value: data.lowRisk },
          ]
        : [],
    [data, t]
  )

  if (isLoading || !data) return <div>{t('fraud.loading')}</div>

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>{t('fraud.title')}</h2>

      {/* Summary stats */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {[
          { label: t('fraud.stats.total'), value: data.totalAlerts, color: '#4a5568' },
          { label: t('fraud.stats.high_risk'), value: data.highRisk, color: '#e53e3e' },
          { label: t('fraud.stats.medium_risk'), value: data.mediumRisk, color: '#dd6b20' },
          { label: t('fraud.stats.low_risk'), value: data.lowRisk, color: '#38a169' },
        ].map((s) => (
          <div key={s.label} style={statCard}>
            <div style={{ fontSize: 12, color: 'var(--text-muted, #718096)' }}>{s.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Risk score over time */}
        <div style={{ flex: '1 1 320px' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{t('fraud.charts.risk_score')}</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={riskOverTime} {...chartConfig}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Tooltip content={<VirtualizedTooltip formatter={riskTooltipFormatter} />} />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#e53e3e"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Risk distribution pie */}
        <div style={{ flex: '0 0 220px' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{t('fraud.charts.risk_distribution')}</div>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius={70}
                label={false}
                isAnimationActive={false}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Legend iconSize={10} />
              <Tooltip content={<VirtualizedTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent alerts table */}
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{t('fraud.alerts.title')}</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>
              {[t('fraud.alerts.columns.account'), t('fraud.alerts.columns.pattern'), t('fraud.alerts.columns.score'), t('fraud.alerts.columns.detected'), t('fraud.alerts.columns.description')].map((h) => (
                <th key={h} style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.recentAlerts.map((alert) => (
              <tr key={alert.id}>
                <td style={td}><code>{alert.accountId}</code></td>
                <td style={td}>{t(PATTERN_LABELS[alert.pattern])}</td>
                <td style={td}>
                  <span style={{ color: RISK_COLOR(alert.riskScore), fontWeight: 600 }}>
                    {alert.riskScore}
                  </span>
                </td>
                <td style={td}>{new Date(alert.detectedAt).toLocaleString()}</td>
                <td style={td}>{alert.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
})

const statCard: React.CSSProperties = {
  border: '1px solid var(--border-light, #eee)',
  borderRadius: 8,
  padding: '10px 16px',
  background: 'var(--bg-card, #fff)',
  minWidth: 100,
}
const th: React.CSSProperties = { textAlign: 'left', borderBottom: '1px solid var(--border-color, #ddd)', padding: '6px 8px' }
const td: React.CSSProperties = { borderBottom: '1px solid var(--border-light, #f1f1f1)', padding: '6px 8px' }