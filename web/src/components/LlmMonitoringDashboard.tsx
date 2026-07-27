import React, { useState, useEffect } from 'react'
import { get } from '../api/client'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts'

interface MetricsData {
  requests_per_min: number
  tokens_per_sec: number
  p50_latency: number
  p95_latency: number
  p99_latency: number
  success_rate: number
  error_rate: number
  daily_spend: number
  weekly_spend: number
  monthly_spend: number
  cost_by_feature: Record<string, number>
  cost_by_model: Record<string, number>
  projected_monthly_cost: number
  cache_hit_rate: number
  avg_eval_score: number
  avg_hallucination_rate: number
  avg_feedback: number
  guarded_requests: number
  safety_incidents: number
  false_positive_rate: number
}

interface Alert {
  type: string
  message: string
  timestamp: number
  status: string
}

export function LlmMonitoringDashboard() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    try {
      const data = await get<MetricsData>('/api/v1/llm-monitoring/metrics')
      const activeAlerts = await get<Alert[]>('/api/v1/llm-monitoring/alerts')
      setMetrics(data)
      setAlerts(activeAlerts)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch LLM metrics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Update every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: '#888' }}>
        <h3>Loading LLM Observability Metrics...</h3>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: 24, color: '#ef4444', textAlign: 'center' }}>
        <h3>Error: {error}</h3>
        <button onClick={fetchMetrics} style={{ marginTop: 12, padding: '8px 16px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}>Retry</button>
      </div>
    )
  }

  if (!metrics) return null

  // Chart data formatting
  const modelCostData = Object.entries(metrics.cost_by_model).map(([name, cost]) => ({ name, cost }))
  const featureCostData = Object.entries(metrics.cost_by_feature).map(([name, cost]) => ({ name, cost }))
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

  return (
    <div style={{
      padding: 24,
      background: 'var(--bg-card, #1e293b)',
      borderRadius: 12,
      boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
      color: 'var(--text-primary, #f8fafc)',
      fontFamily: 'Inter, system-ui, sans-serif'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>LLM Observability & Monitoring Dashboard</h2>
          <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: 14 }}>Real-time metrics, cost forecasting, and safety guardrails</p>
        </div>
        <button
          onClick={fetchMetrics}
          style={{
            background: '#3b82f6',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '8px 16px',
            fontSize: 14,
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'background 0.2s',
          }}
          onMouseOver={(e) => (e.currentTarget.style.background = '#2563eb')}
          onMouseOut={(e) => (e.currentTarget.style.background = '#3b82f6')}
        >
          Refresh Data
        </button>
      </div>

      {/* Alert Banner if Active Alerts exist */}
      {alerts.length > 0 && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', borderLeft: '4px solid #ef4444', padding: 16, borderRadius: 6, marginBottom: 24 }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#f87171', fontWeight: 600 }}>Active Alerts / Anomalies</h4>
          <ul style={{ margin: 0, paddingLeft: 20, color: '#fca5a5', fontSize: 14 }}>
            {alerts.map((alert, idx) => (
              <li key={idx} style={{ marginBottom: 4 }}>{alert.message}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Quick Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
        <div style={{ background: '#334155', padding: 16, borderRadius: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>Requests / Min</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{metrics.requests_per_min}</div>
        </div>
        <div style={{ background: '#334155', padding: 16, borderRadius: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>Tokens / Sec</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{metrics.tokens_per_sec.toFixed(2)}</div>
        </div>
        <div style={{ background: '#334155', padding: 16, borderRadius: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>P95 Latency</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4, color: metrics.p95_latency > 2.0 ? '#ef4444' : '#f8fafc' }}>
            {metrics.p95_latency.toFixed(2)}s
          </div>
        </div>
        <div style={{ background: '#334155', padding: 16, borderRadius: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>Monthly Spend</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>${metrics.monthly_spend.toFixed(2)}</div>
        </div>
        <div style={{ background: '#334155', padding: 16, borderRadius: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>Cache Hit Rate</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4, color: '#10b981' }}>
            {(metrics.cache_hit_rate * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Main Charts & Details Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24, marginBottom: 24 }}>
        {/* Cost per Model */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', padding: 20, borderRadius: 8 }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: 16, fontWeight: 600 }}>Cost per Model</h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelCostData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Bar dataKey="cost" fill="#3b82f6">
                  {modelCostData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cost per Feature */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', padding: 20, borderRadius: 8 }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: 16, fontWeight: 600 }}>Cost per Feature</h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={featureCostData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  outerRadius={70}
                  fill="#8884d8"
                  dataKey="cost"
                >
                  {featureCostData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16 }}>
        {/* Performance & Quality details */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', padding: 16, borderRadius: 8 }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600 }}>Performance & Quality</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Success Rate:</span>
              <span style={{ fontWeight: 600 }}>{(metrics.success_rate * 100).toFixed(1)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Avg Eval Score:</span>
              <span style={{ fontWeight: 600 }}>{metrics.avg_eval_score.toFixed(2)} / 1.00</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Hallucination Rate:</span>
              <span style={{ fontWeight: 600, color: metrics.avg_hallucination_rate > 0.05 ? '#ef4444' : '#f8fafc' }}>
                {(metrics.avg_hallucination_rate * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>User Feedback:</span>
              <span style={{ fontWeight: 600 }}>{metrics.avg_feedback.toFixed(1)} / 5.0</span>
            </div>
          </div>
        </div>

        {/* Safety Guardrails */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', padding: 16, borderRadius: 8 }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600 }}>Safety Guardrails</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Guarded Requests:</span>
              <span>{metrics.guarded_requests}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Safety Incidents:</span>
              <span style={{ fontWeight: 600, color: metrics.safety_incidents > 0 ? '#ef4444' : '#10b981' }}>
                {metrics.safety_incidents}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>False Positive Rate:</span>
              <span>{(metrics.false_positive_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Cost Projections */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', padding: 16, borderRadius: 8 }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600 }}>Cost Forecasting</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Daily Avg Spend:</span>
              <span>${metrics.daily_spend.toFixed(2)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Projected Monthly:</span>
              <span style={{ fontWeight: 600, color: metrics.projected_monthly_cost > metrics.daily_spend * 30 ? '#f59e0b' : '#f8fafc' }}>
                ${metrics.projected_monthly_cost.toFixed(2)}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Forecast Accuracy:</span>
              <span>&plusmn;10%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
export default LlmMonitoringDashboard
