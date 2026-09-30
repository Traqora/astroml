import { memo, useMemo, useState, useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { get } from '../../api/client'
import { ApiError } from '../../api/client'
import { VirtualizedTooltip } from '../charts/VirtualizedTooltip'
import { createChartConfig, sampleData, CHART_TARGET_POINTS } from '../../lib/chartUtils'
import { SkeletonModelMonitoring } from '../Skeletons'
import { useWebSocket } from '../../hooks/useWebSocket'
import { ModelSelector } from './ModelSelector'
import { ExportButton } from './ExportButton'
import { ModelComparisonChart } from './ModelComparisonChart'
import { LastUpdated } from './LastUpdated'

interface MonitoringMetrics {
  accuracy: number
  f1: number
  drift_score: number
  auc: number
}

interface PerformancePoint {
  date: string
  accuracy: number
  drift: number
}

interface MonitoringResponse {
  metrics: MonitoringMetrics
  performance: PerformancePoint[]
  model?: string
  timestamp?: string
}

interface ModelRun {
  id: string
  name: string
  color: string
  data: PerformancePoint[]
}

async function getMonitoringData(model?: string): Promise<MonitoringResponse> {
  try {
    const endpoint = model
      ? `/api/v1/monitoring/metrics?model=${model}`
      : '/api/v1/monitoring/metrics'
    const response = await get<any>(endpoint)

    return {
      metrics: {
        accuracy: response.accuracy || 0,
        f1: response.f1 || 0,
        drift_score: response.drift_score || 0,
        auc: response.auc || 0,
      },
      performance: response.performance || [],
      model: response.model || 'default',
      timestamp: response.timestamp || new Date().toISOString(),
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return {
        metrics: {
          accuracy: 0.93,
          f1: 0.86,
          drift_score: 0.12,
          auc: 0.91,
        },
        performance: [
          { date: '2026-04-01', accuracy: 0.88, drift: 0.08 },
          { date: '2026-04-08', accuracy: 0.91, drift: 0.10 },
          { date: '2026-04-15', accuracy: 0.90, drift: 0.12 },
          { date: '2026-04-22', accuracy: 0.92, drift: 0.09 },
          { date: '2026-04-29', accuracy: 0.93, drift: 0.07 },
        ],
        model: 'default',
        timestamp: new Date().toISOString(),
      }
    }
    throw error
  }
}

// Available models for comparison
const AVAILABLE_MODELS = ['Production', 'Candidate v1', 'Candidate v2', 'Baseline']
const MODEL_COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#eab308']

const chartConfig = createChartConfig()
const accuracyFormatter = (value: number) => `${(value * 100).toFixed(1)}%`
const driftFormatter = (value: number) => value.toFixed(2)

export const ModelMonitoringDashboard = memo(function ModelMonitoringDashboard() {
  const { t } = useTranslation()
  const [selectedModel, setSelectedModel] = useState('Production')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [isRealtime, setIsRealtime] = useState(false)
  const [comparisonModels, setComparisonModels] = useState<string[]>(['Production', 'Candidate v1'])

  // WebSocket connection for real-time updates
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  const wsUrl = (import.meta.env.VITE_WS_URL as string | undefined)
    || `${apiBase.replace(/^http/, 'ws')}/api/v1/ws/metrics`

  const { lastMessage, isConnected } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data.type === 'metrics_update') {
        // Update the dashboard with new metrics
        setLastUpdated(new Date())
        setIsRealtime(true)
        // Refetch data to get latest metrics
        refetch()
      }
    },
    onError: (error) => {
      console.error('WebSocket error:', error)
    },
  })

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['monitoring', selectedModel],
    queryFn: () => getMonitoringData(selectedModel),
    refetchInterval: isConnected ? false : 30000, // Disable polling when WebSocket is connected
  })

  // Fetch comparison data for multiple models
  const comparisonData = useQuery({
    queryKey: ['monitoring-comparison', comparisonModels],
    queryFn: async () => {
      const results = await Promise.all(
        comparisonModels.map((model, index) =>
          getMonitoringData(model).then((res) => ({
            id: model,
            name: model,
            color: MODEL_COLORS[index % MODEL_COLORS.length],
            data: res.performance,
          }))
        )
      )
      return results
    },
    enabled: comparisonModels.length > 1,
  })

  // Downsample performance series — it may grow large in long-running deployments
  const performanceData = useMemo(
    () => (data ? sampleData(data.performance, CHART_TARGET_POINTS, 'accuracy') : []),
    [data]
  )

  // Handle WebSocket message for real-time updates
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'metrics_update') {
      setLastUpdated(new Date())
      setIsRealtime(true)
    }
  }, [lastMessage])

  const handleModelChange = useCallback((model: string) => {
    setSelectedModel(model)
  }, [])

  const handleComparisonToggle = useCallback((model: string) => {
    setComparisonModels((prev) => {
      if (prev.includes(model)) {
        return prev.filter((m) => m !== model)
      }
      return [...prev, model]
    })
  }, [])

  if (isLoading) return <SkeletonModelMonitoring />
  if (error) return <div>{t('monitoring.errors.loading', { message: (error as Error).message })}</div>
  if (!data) return <div>{t('monitoring.errors.no_data')}</div>

  const metrics = [
    { label: t('monitoring.metrics.accuracy'), value: `${(data.metrics.accuracy * 100).toFixed(1)}%`, description: t('monitoring.metrics.accuracy_desc') },
    { label: t('monitoring.metrics.f1'), value: data.metrics.f1.toFixed(2), description: t('monitoring.metrics.f1_desc') },
    { label: t('monitoring.metrics.drift'), value: data.metrics.drift_score.toFixed(2), description: t('monitoring.metrics.drift_desc') },
    { label: t('monitoring.metrics.auc'), value: data.metrics.auc.toFixed(2), description: t('monitoring.metrics.auc_desc') },
  ]

  return (
    <section style={{ display: 'grid', gap: 24 }}>
      {/* Header with controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            models={AVAILABLE_MODELS}
          />
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {AVAILABLE_MODELS.map((model) => (
              <label
                key={model}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 13,
                  color: 'var(--text-secondary, #555)',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={comparisonModels.includes(model)}
                  onChange={() => handleComparisonToggle(model)}
                  disabled={model === selectedModel}
                />
                {model}
              </label>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <LastUpdated timestamp={lastUpdated || new Date()} isRealtime={isRealtime} />
          <ExportButton data={data} filename={`model-metrics-${selectedModel}`} />
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        {metrics.map((metric) => (
          <div
            key={metric.label}
            style={{
              padding: 20,
              borderRadius: 16,
              background: 'var(--bg-card, #fff)',
              boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0, 0, 0, 0.06))',
              border: '1px solid var(--card-border, #ececec)',
            }}
          >
            <p style={{ margin: 0, fontSize: 14, color: 'var(--text-secondary, #666)' }}>{metric.label}</p>
            <p style={{ margin: '12px 0', fontSize: 28, fontWeight: 700 }}>{metric.value}</p>
            <p style={{ margin: 0, fontSize: 12, color: 'var(--text-muted, #888)' }}>{metric.description}</p>
          </div>
        ))}
      </div>

      {/* Charts grid */}
      <div style={{ display: 'grid', gap: 24, gridTemplateColumns: '1.5fr 1fr' }}>
        {/* Accuracy trend with comparison */}
        <div
          style={{
            minHeight: 320,
            padding: 20,
            borderRadius: 16,
            background: 'var(--bg-card, #fff)',
            boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0, 0, 0, 0.06))',
            border: '1px solid var(--card-border, #ececec)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <h2 style={{ margin: 0 }}>{t('monitoring.charts.accuracy_trend')}</h2>
            {comparisonModels.length > 1 && (
              <span style={{ fontSize: 12, color: 'var(--text-muted, #888)' }}>
                {t('monitoring.comparison.active')}: {comparisonModels.length}
              </span>
            )}
          </div>
          {comparisonModels.length > 1 && comparisonData.data ? (
            <ModelComparisonChart runs={comparisonData.data} height={240} />
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={performanceData} {...chartConfig}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color, #f0f0f0)" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis domain={[0.7, 1.0]} tickFormatter={accuracyFormatter} />
                <Tooltip content={<VirtualizedTooltip formatter={accuracyFormatter} />} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#3f8efc"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Drift detection */}
        <div
          style={{
            minHeight: 320,
            padding: 20,
            borderRadius: 16,
            background: 'var(--bg-card, #fff)',
            boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0, 0, 0, 0.06))',
            border: '1px solid var(--card-border, #ececec)',
          }}
        >
          <h2 style={{ marginTop: 0 }}>{t('monitoring.charts.drift_detection')}</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={performanceData} {...chartConfig}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color, #f0f0f0)" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tickFormatter={driftFormatter} />
              <Tooltip content={<VirtualizedTooltip formatter={driftFormatter} />} />
              <Legend />
              <Bar dataKey="drift" fill="#f65d5d" radius={[8, 8, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <p style={{ marginTop: 12, fontSize: 14, color: 'var(--text-secondary, #555)' }}>
            {t('monitoring.charts.drift_description')}
          </p>
        </div>
      </div>

      {/* Real-time status indicator */}
      {isConnected && (
        <div
          style={{
            padding: 12,
            borderRadius: 8,
            background: '#ecfdf5',
            border: '1px solid #6ee7b7',
            color: '#065f46',
            fontSize: 14,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span
            style={{
              display: 'inline-block',
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: '#22c55e',
              animation: 'pulse 2s infinite',
            }}
          />
          {t('monitoring.realtime.active')}
        </div>
      )}
    </section>
  )
})