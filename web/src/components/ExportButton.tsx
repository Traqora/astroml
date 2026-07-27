import { useState } from 'react'
import {
  exportData,
  exportLargeData,
  type ExportFormat,
  type ExportDataType,
  type ExportProgress,
} from '../lib/export'

interface ExportButtonProps {
  dataType: ExportDataType
  format?: ExportFormat
  filters?: Record<string, string | number | boolean | undefined>
  filename?: string
  large?: boolean
  label?: string
}

export function ExportButton({
  dataType,
  format = 'csv',
  filters,
  filename,
  large = false,
  label,
}: ExportButtonProps) {
  const [exporting, setExporting] = useState(false)
  const [progress, setProgress] = useState<ExportProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleExport = async () => {
    setExporting(true)
    setProgress(null)
    setError(null)

    const options = { format, dataType, filters, filename }

    try {
      if (large) {
        await exportLargeData(options, {
          onProgress: setProgress,
          onError: (err) => setError(err.message),
        })
      } else {
        await exportData(options, {
          onProgress: setProgress,
          onError: (err) => setError(err.message),
        })
      }
    } finally {
      setExporting(false)
      setProgress(null)
    }
  }

  const formatLabel: string = format.toUpperCase()
  const btnLabel = label || `Export ${formatLabel}`

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <button
        onClick={handleExport}
        disabled={exporting}
        style={{
          padding: '6px 14px',
          borderRadius: 6,
          border: '1px solid var(--border-color, #ddd)',
          background: 'var(--bg-card, #fff)',
          color: 'var(--text-primary, #1a202c)',
          cursor: exporting ? 'not-allowed' : 'pointer',
          fontSize: 13,
          fontWeight: 600,
          opacity: exporting ? 0.6 : 1,
        }}
      >
        {exporting ? 'Exporting...' : btnLabel}
      </button>

      {progress && (
        <div style={{ fontSize: 12, color: 'var(--text-muted, #718096)' }}>
          {progress.percentage}% ({progress.loaded}/{progress.total})
        </div>
      )}

      {error && (
        <div style={{ fontSize: 12, color: '#e53e3e' }}>
          {error}
        </div>
      )}
    </div>
  )
}

export function ExportToolbar({
  dataType,
  filters,
  large,
}: {
  dataType: ExportDataType
  filters?: Record<string, string | number | boolean | undefined>
  large?: boolean
}) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <ExportButton dataType={dataType} format="csv" filters={filters} large={large} />
      <ExportButton dataType={dataType} format="json" filters={filters} large={large} />
      <ExportButton dataType={dataType} format="parquet" filters={filters} large={large} />
    </div>
  )
}
