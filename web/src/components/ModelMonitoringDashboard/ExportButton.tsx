import { memo } from 'react'
import { useTranslation } from 'react-i18next'

interface ExportButtonProps {
  data: any
  filename?: string
}

export const ExportButton = memo(function ExportButton({
  data,
  filename = 'model-metrics',
}: ExportButtonProps) {
  const { t } = useTranslation()

  const exportCSV = () => {
    if (!data) return

    // Prepare CSV data
    const headers = ['Timestamp', 'Accuracy', 'F1 Score', 'Drift Score', 'AUC']
    const rows = [
      headers.join(','),
      [
        new Date().toISOString(),
        data.metrics.accuracy,
        data.metrics.f1,
        data.metrics.drift_score,
        data.metrics.auc,
      ].join(','),
    ]

    // Add performance history if available
    if (data.performance && data.performance.length > 0) {
      rows.push('')
      rows.push('History,Accuracy,Drift')
      data.performance.forEach((point: any) => {
        rows.push([point.date, point.accuracy, point.drift].join(','))
      })
    }

    const csv = rows.join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${filename}-${new Date().toISOString().slice(0, 10)}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  return (
    <button
      onClick={exportCSV}
      style={{
        padding: '8px 16px',
        borderRadius: 6,
        border: '1px solid #d1d5db',
        background: '#fff',
        cursor: 'pointer',
        fontSize: 14,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        transition: 'all 0.2s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f9fafb'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = '#fff'
      }}
    >
      <span>📊</span>
      {t('monitoring.export.button')}
    </button>
  )
})