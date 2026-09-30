import { useTranslation } from 'react-i18next'
import type { PointsHistoryResponse } from '../../lib/types'

export function PointsHistoryTable({
  response,
  loading,
  page,
  pageSize,
  onPageChange,
}: {
  response: PointsHistoryResponse | undefined
  loading: boolean
  page: number
  pageSize: number
  onPageChange: (p: number) => void
}) {
  const { t } = useTranslation()
  const total = response?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: '8px 0' }}>{t('loyalty.history.title')}</h2>
        <div>
          <button disabled={page === 0 || loading} onClick={() => onPageChange(page - 1)}>{t('transactions.table.prev')}</button>
          <span style={{ margin: '0 8px' }}>{t('transactions.table.page', { current: page + 1, total: totalPages })}</span>
          <button disabled={page + 1 >= totalPages || loading} onClick={() => onPageChange(page + 1)}>{t('transactions.table.next')}</button>
        </div>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={th}>{t('loyalty.history.columns.date')}</th>
              <th style={th}>{t('loyalty.history.columns.type')}</th>
              <th style={th}>{t('loyalty.history.columns.points')}</th>
              <th style={th}>{t('loyalty.history.columns.source')}</th>
              <th style={th}>{t('loyalty.history.columns.note')}</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={5} style={{ padding: 12, textAlign: 'center' }}>{t('loyalty.history.loading')}</td></tr>
            )}
            {!loading && response?.data.length === 0 && (
              <tr><td colSpan={5} style={{ padding: 12, textAlign: 'center' }}>{t('loyalty.history.no_data')}</td></tr>
            )}
            {!loading && response?.data.map((t) => (
              <tr key={t.id}>
                <td style={td}>{new Date(t.date).toLocaleDateString()}</td>
                <td style={td}>{t.type}</td>
                <td style={td}>{t.points}</td>
                <td style={td}>{t.source ?? '-'}</td>
                <td style={td}>{t.note ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const th: React.CSSProperties = { textAlign: 'left', borderBottom: '1px solid var(--border-color, #ddd)', padding: 8 }
const td: React.CSSProperties = { borderBottom: '1px solid var(--border-light, #f1f1f1)', padding: 8 }