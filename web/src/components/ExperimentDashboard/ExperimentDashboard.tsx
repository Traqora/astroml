import { memo, useMemo, useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
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
import { get, post, del, patch } from '../../api/client'
import { createChartConfig, sampleData, CHART_TARGET_POINTS } from '../../lib/chartUtils'
import { SkeletonTable } from '../Skeletons/SkeletonTable'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ExperimentItem {
  experiment_id: string
  name: string
  description: string
  tags: string[]
  notes: string
  created_at: string
  updated_at: string
  run_count: number
}

interface MetricSummary {
  min: number
  max: number
  mean: number
}

interface ExperimentReport {
  experiment_id: string
  experiment_name: string
  generated_at: string
  num_runs: number
  best_run: Record<string, unknown> | null
  metric_summary: Record<string, MetricSummary>
  param_importance: Array<{ parameter: string; importance: number }>
  notes: string
}

interface DashboardStats {
  total_experiments: number
  total_runs: number
  tag_counts: Record<string, number>
  recent_experiments: Array<{ id: string; name: string; runs: number; tags: string[]; updated_at: string }>
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const chartConfig = createChartConfig()

async function fetchExperiments(params?: {
  tag?: string[]
  search?: string
  sort_by?: string
  reverse?: boolean
}): Promise<ExperimentItem[]> {
  const query = new URLSearchParams()
  if (params?.tag) params.tag.forEach((t) => query.append('tag', t))
  if (params?.search) query.set('search', params.search)
  if (params?.sort_by) query.set('sort_by', params.sort_by)
  if (params?.reverse) query.set('reverse', String(params.reverse))
  const qs = query.toString()
  return get<ExperimentItem[]>(`/api/v1/experiments${qs ? `?${qs}` : ''}`)
}

async function fetchReport(experimentId: string): Promise<ExperimentReport> {
  return get<ExperimentReport>(`/api/v1/experiments/${experimentId}/report`)
}

async function fetchStats(): Promise<DashboardStats> {
  return get<DashboardStats>('/api/v1/experiments/stats')
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ExperimentDashboard = memo(function ExperimentDashboard() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)

  // Fetch experiments list
  const {
    data: experiments,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['experiments', { search: searchQuery, tags: selectedTags }],
    queryFn: () =>
      fetchExperiments({
        search: searchQuery || undefined,
        tag: selectedTags.length > 0 ? selectedTags : undefined,
      }),
  })

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['experiments-stats'],
    queryFn: fetchStats,
  })

  // Fetch report for selected experiment
  const { data: report, isLoading: reportLoading } = useQuery({
    queryKey: ['experiment-report', selectedExperiment],
    queryFn: () => fetchReport(selectedExperiment!),
    enabled: !!selectedExperiment,
  })

  // Create experiment mutation
  const createMutation = useMutation({
    mutationFn: (data: { name: string; description: string; tags: string[]; notes: string }) =>
      post('/api/v1/experiments', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      queryClient.invalidateQueries({ queryKey: ['experiments-stats'] })
    },
  })

  // Delete experiment mutation
  const deleteMutation = useMutation({
    mutationFn: (experimentId: string) => del(`/api/v1/experiments/${experimentId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      queryClient.invalidateQueries({ queryKey: ['experiments-stats'] })
      if (selectedExperiment) setSelectedExperiment(null)
    },
  })

  // Add tag mutation
  const addTagMutation = useMutation({
    mutationFn: ({ id, tag }: { id: string; tag: string }) =>
      post(`/api/v1/experiments/${id}/tags?tag=${encodeURIComponent(tag)}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
    },
  })

  const handleCreate = useCallback(async () => {
    const name = prompt(t('experiments.create.prompt_name', 'Experiment name:'))
    if (!name) return
    const desc = prompt(t('experiments.create.prompt_desc', 'Description (optional):')) || ''
    await createMutation.mutateAsync({ name, description: desc, tags: [], notes: '' })
    refetch()
  }, [createMutation, refetch, t])

  const handleDelete = useCallback(
    async (id: string) => {
      if (!confirm(t('experiments.delete.confirm', 'Delete this experiment?'))) return
      await deleteMutation.mutateAsync(id)
    },
    [deleteMutation, t]
  )

  const handleAddTag = useCallback(
    async (id: string) => {
      const tag = prompt(t('experiments.tag.prompt', 'Tag name:'))
      if (!tag) return
      await addTagMutation.mutateAsync({ id, tag })
    },
    [addTagMutation, t]
  )

  // Collect all unique tags
  const allTags = useMemo(() => {
    if (!experiments) return []
    const tagSet = new Set<string>()
    experiments.forEach((e) => e.tags.forEach((t) => tagSet.add(t)))
    return Array.from(tagSet).sort()
  }, [experiments])

  // Toggle tag filter
  const toggleTagFilter = useCallback(
    (tag: string) => {
      setSelectedTags((prev) =>
        prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
      )
    },
    []
  )

  if (isLoading) return <SkeletonTable rows={5} cols={5} />

  return (
    <section style={{ display: 'grid', gap: 24 }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <div>
          <h1 style={{ margin: 0 }}>{t('experiments.title', 'Experiment Tracking')}</h1>
          {stats && (
            <p style={{ margin: '4px 0 0', fontSize: 14, color: 'var(--text-secondary, #666)' }}>
              {stats.total_experiments} experiments, {stats.total_runs} runs
            </p>
          )}
        </div>
        <button
          onClick={handleCreate}
          style={{
            padding: '10px 20px',
            borderRadius: 8,
            background: 'var(--accent, #3b82f6)',
            color: '#fff',
            border: 'none',
            cursor: 'pointer',
            fontWeight: 600,
          }}
        >
          + {t('experiments.create.button', 'New Experiment')}
        </button>
      </div>

      {/* Search and tag filters */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          type="text"
          placeholder={t('experiments.search.placeholder', 'Search experiments...')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: '8px 14px',
            borderRadius: 8,
            border: '1px solid var(--card-border, #ddd)',
            fontSize: 14,
            flex: '1 1 200px',
            maxWidth: 320,
          }}
        />
        {allTags.map((tag) => (
          <button
            key={tag}
            onClick={() => toggleTagFilter(tag)}
            style={{
              padding: '4px 12px',
              borderRadius: 20,
              fontSize: 12,
              border: '1px solid var(--card-border, #ddd)',
              background: selectedTags.includes(tag)
                ? 'var(--accent, #3b82f6)'
                : 'var(--bg-card, #fff)',
              color: selectedTags.includes(tag) ? '#fff' : 'var(--text-primary, #333)',
              cursor: 'pointer',
            }}
          >
            {tag}
          </button>
        ))}
      </div>

      {/* Content: left list + right details */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 2fr', gap: 24 }}>
        {/* Experiment list */}
        <div
          style={{
            padding: 16,
            borderRadius: 16,
            background: 'var(--bg-card, #fff)',
            boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0,0,0,0.06))',
            border: '1px solid var(--card-border, #ececec)',
            maxHeight: 600,
            overflow: 'auto',
          }}
        >
          <h3 style={{ marginTop: 0 }}>{t('experiments.list.title', 'Experiments')}</h3>
          {error && (
            <p style={{ color: 'red' }}>
              {t('experiments.errors.loading', { message: (error as Error).message })}
            </p>
          )}
          {experiments && experiments.length === 0 && (
            <p style={{ color: 'var(--text-muted, #888)' }}>
              {t('experiments.empty', 'No experiments yet.')}
            </p>
          )}
          {experiments?.map((exp) => (
            <div
              key={exp.experiment_id}
              onClick={() => setSelectedExperiment(exp.experiment_id)}
              style={{
                padding: '12px 8px',
                borderBottom: '1px solid var(--card-border, #eee)',
                cursor: 'pointer',
                background:
                  selectedExperiment === exp.experiment_id
                    ? 'var(--accent-light, #eff6ff)'
                    : 'transparent',
                borderRadius: 8,
                marginBottom: 4,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ fontSize: 14 }}>{exp.name}</strong>
                <span style={{ fontSize: 12, color: 'var(--text-muted, #888)' }}>
                  {exp.run_count} {t('experiments.runs', 'runs')}
                </span>
              </div>
              <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {exp.tags.map((tag) => (
                  <span
                    key={tag}
                    style={{
                      fontSize: 10,
                      padding: '2px 8px',
                      borderRadius: 10,
                      background: 'var(--bg-muted, #f3f4f6)',
                      color: 'var(--text-secondary, #666)',
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <div style={{ marginTop: 6, display: 'flex', gap: 8 }}>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleAddTag(exp.experiment_id)
                  }}
                  style={{ fontSize: 11, padding: '2px 8px', cursor: 'pointer' }}
                >
                  {t('experiments.tag.add', '+Tag')}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(exp.experiment_id)
                  }}
                  style={{ fontSize: 11, padding: '2px 8px', cursor: 'pointer', color: '#ef4444' }}
                >
                  {t('experiments.delete.button', 'Delete')}
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Report / details panel */}
        <div
          style={{
            padding: 20,
            borderRadius: 16,
            background: 'var(--bg-card, #fff)',
            boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0,0,0,0.06))',
            border: '1px solid var(--card-border, #ececec)',
            minHeight: 400,
            maxHeight: 600,
            overflow: 'auto',
          }}
        >
          {!selectedExperiment && (
            <p style={{ color: 'var(--text-muted, #888)' }}>
              {t('experiments.report.select', 'Select an experiment to view its report.')}
            </p>
          )}
          {selectedExperiment && reportLoading && <p>{t('common.loading', 'Loading...')}</p>}
          {selectedExperiment && report && (
            <div>
              <h2 style={{ marginTop: 0 }}>{report.experiment_name}</h2>
              <p style={{ fontSize: 12, color: 'var(--text-muted, #888)' }}>
                {t('experiments.report.generated', 'Report generated')}: {report.generated_at}
              </p>

              {/* Best run */}
              {report.best_run && (
                <div style={{ marginTop: 16 }}>
                  <h3>{t('experiments.report.best_run', 'Best Run')}</h3>
                  <div
                    style={{
                      padding: 12,
                      borderRadius: 8,
                      background: '#ecfdf5',
                      border: '1px solid #6ee7b7',
                    }}
                  >
                    <p style={{ margin: 0 }}>
                      <strong>{(report.best_run as Record<string, unknown>).run_name as string}</strong>
                    </p>
                    <p style={{ margin: '4px 0 0', fontSize: 13, color: '#065f46' }}>
                      Status: {(report.best_run as Record<string, unknown>).status as string}
                    </p>
                  </div>
                </div>
              )}

              {/* Metric summary */}
              {Object.keys(report.metric_summary).length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h3>{t('experiments.report.metrics', 'Metric Summary')}</h3>
                  <div style={{ overflow: 'auto' }}>
                    <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', padding: 4 }}>{t('experiments.metrics.name', 'Metric')}</th>
                          <th style={{ textAlign: 'right', padding: 4 }}>{t('experiments.metrics.min', 'Min')}</th>
                          <th style={{ textAlign: 'right', padding: 4 }}>{t('experiments.metrics.max', 'Max')}</th>
                          <th style={{ textAlign: 'right', padding: 4 }}>{t('experiments.metrics.mean', 'Mean')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(report.metric_summary).map(([metric, s]) => (
                          <tr key={metric}>
                            <td style={{ padding: 4 }}>{metric}</td>
                            <td style={{ textAlign: 'right', padding: 4 }}>{s.min.toFixed(4)}</td>
                            <td style={{ textAlign: 'right', padding: 4 }}>{s.max.toFixed(4)}</td>
                            <td style={{ textAlign: 'right', padding: 4 }}>{s.mean.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Hyperparameter importance chart */}
              {report.param_importance.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h3>{t('experiments.report.param_importance', 'Hyperparameter Importance')}</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart
                      data={report.param_importance.map((p) => ({
                        name: p.parameter,
                        importance: p.importance,
                      }))}
                      layout="vertical"
                      {...chartConfig}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color, #f0f0f0)" />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Notes */}
              {report.notes && (
                <div style={{ marginTop: 16 }}>
                  <h3>{t('experiments.report.notes', 'Notes')}</h3>
                  <p style={{ fontSize: 13, color: 'var(--text-secondary, #555)', whiteSpace: 'pre-wrap' }}>
                    {report.notes}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
})