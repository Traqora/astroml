import { apiRequest } from '../api/client'

export type ExportFormat = 'csv' | 'json' | 'parquet'

export type ExportDataType = 'transactions' | 'graph' | 'predictions'

export interface ExportOptions {
  format: ExportFormat
  dataType: ExportDataType
  filters?: Record<string, string | number | boolean | undefined>
  filename?: string
}

export interface ExportProgress {
  loaded: number
  total: number
  percentage: number
}

export interface ExportCallbacks {
  onProgress?: (progress: ExportProgress) => void
  onComplete?: (blob: Blob) => void
  onError?: (error: Error) => void
}

function formatDate(): string {
  return new Date().toISOString().slice(0, 10)
}

function getMimeType(format: ExportFormat): string {
  switch (format) {
    case 'csv': return 'text/csv'
    case 'json': return 'application/json'
    case 'parquet': return 'application/octet-stream'
  }
}

function getExtension(format: ExportFormat): string {
  switch (format) {
    case 'csv': return 'csv'
    case 'json': return 'json'
    case 'parquet': return 'parquet'
  }
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export async function fetchExportData(
  dataType: ExportDataType,
  filters?: Record<string, string | number | boolean | undefined>,
  signal?: AbortSignal
): Promise<any[]> {
  const params = new URLSearchParams()

  if (filters) {
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== '') {
        params.append(key, String(value))
      }
    })
  }

  const endpointMap: Record<ExportDataType, string> = {
    transactions: '/api/v1/transactions',
    graph: '/api/v1/graph',
    predictions: '/api/v1/predictions',
  }

  const endpoint = `${endpointMap[dataType]}?${params.toString()}`
  const response = await apiRequest<{ data: any[] }>(endpoint, { signal })
  return response.data
}

export async function exportData(
  options: ExportOptions,
  callbacks?: ExportCallbacks
): Promise<void> {
  const { format, dataType, filters, filename } = options
  const { onProgress, onComplete, onError } = callbacks || {}

  try {
    const data = await fetchExportData(dataType, filters)

    if (onProgress) {
      onProgress({ loaded: data.length, total: data.length, percentage: 100 })
    }

    const mimeType = getMimeType(format)
    const ext = getExtension(format)
    const defaultName = `${dataType}-${formatDate()}.${ext}`
    const name = filename || defaultName

    let blob: Blob

    switch (format) {
      case 'csv':
        blob = exportToCsv(data, mimeType)
        break
      case 'json':
        blob = exportToJson(data, mimeType)
        break
      case 'parquet':
        blob = exportToParquet(data, mimeType)
        break
    }

    onComplete?.(blob)
    downloadBlob(blob, name)
  } catch (error) {
    onError?.(error instanceof Error ? error : new Error(String(error)))
  }
}

export function exportToCsv(data: any[], mimeType: string = 'text/csv'): Blob {
  if (data.length === 0) {
    return new Blob([''], { type: mimeType })
  }

  const headers = Object.keys(data[0])
  const csvRows: string[] = []

  csvRows.push(headers.join(','))

  for (const row of data) {
    const values = headers.map((header) => {
      const val = row[header]
      if (val === null || val === undefined) return ''
      const escaped = String(val).replace(/"/g, '""')
      return `"${escaped}"`
    })
    csvRows.push(values.join(','))
  }

  return new Blob([csvRows.join('\n')], { type: mimeType })
}

export function exportToJson(data: any[], mimeType: string = 'application/json'): Blob {
  const json = JSON.stringify(data, null, 2)
  return new Blob([json], { type: mimeType })
}

export function exportToParquet(data: any[], mimeType: string = 'application/octet-stream'): Blob {
  const json = JSON.stringify(data)
  return new Blob([json], { type: mimeType })
}

export async function exportLargeData(
  options: ExportOptions,
  callbacks?: ExportCallbacks,
  batchSize: number = 1000
): Promise<void> {
  const { format, dataType, filters, filename } = options
  const { onProgress, onComplete, onError } = callbacks || {}

  try {
    const allData: any[] = []
    let page = 0
    let total = 0

    const fetchBatch = async (): Promise<boolean> => {
      const batchFilters = { ...filters, page: page.toString(), page_size: batchSize.toString() } as any
      const params = new URLSearchParams()

      Object.entries(batchFilters).forEach(([key, value]) => {
        if (value !== undefined && value !== '') {
          params.append(key, String(value))
        }
      })

      const endpointMap: Record<ExportDataType, string> = {
        transactions: '/api/v1/transactions',
        graph: '/api/v1/graph',
        predictions: '/api/v1/predictions',
      }

      const endpoint = `${endpointMap[dataType]}?${params.toString()}`
      const response = await apiRequest<{ data: any[]; total?: number }>(endpoint)
      const batch = response.data

      if (!batch || batch.length === 0) return false

      allData.push(...batch)
      total = response.total || allData.length

      onProgress?.({
        loaded: allData.length,
        total,
        percentage: Math.round((allData.length / total) * 100),
      })

      page++
      return batch.length === batchSize
    }

    let hasMore = true
    while (hasMore) {
      hasMore = await fetchBatch()
    }

    const mimeType = getMimeType(format)
    const ext = getExtension(format)
    const defaultName = `${dataType}-${formatDate()}.${ext}`
    const name = filename || defaultName

    let blob: Blob
    switch (format) {
      case 'csv':
        blob = exportToCsv(allData, mimeType)
        break
      case 'json':
        blob = exportToJson(allData, mimeType)
        break
      case 'parquet':
        blob = exportToParquet(allData, mimeType)
        break
    }

    onComplete?.(blob)
    downloadBlob(blob, name)
  } catch (error) {
    onError?.(error instanceof Error ? error : new Error(String(error)))
  }
}
