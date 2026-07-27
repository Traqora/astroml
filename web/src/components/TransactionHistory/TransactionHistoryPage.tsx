import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useTransactionHistory } from '../../hooks/useTransactionHistory'
import { TransactionHistoryTable } from './TransactionHistoryTable'
import { SkeletonTransactionHistory } from '../Skeletons'

export function TransactionHistoryPage() {
  const { t } = useTranslation()
  const [page, setPage] = useState(0)
  const pageSize = 20
  
  const [filters, setFilters] = useState<{
    sourceAccount?: string
    operationType?: string
    startDate?: string
    endDate?: string
  }>({})

  const { data: history, isLoading: loading } = useTransactionHistory(page, pageSize, filters)

  const handleFilterChange = (key: string, value: string) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value || undefined,
    }))
    setPage(0) // Reset to first page when filters change
  }

  if (loading) return <SkeletonTransactionHistory />

  return (
    <div style={{ display: 'grid', gap: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
        <div>
          <h1 style={{ margin: '0 0 16px 0', fontSize: 28, fontWeight: 700 }}>{t('transactions.title')}</h1>
          <p style={{ margin: 0, color: 'var(--text-muted, #666)' }}>
            {t('transactions.subtitle')}
          </p>
        </div>
      </div>

      <div style={{
        padding: 16,
        backgroundColor: 'var(--bg-secondary, #f9f9f9)',
        borderRadius: 8,
        border: '1px solid var(--border-color, #e0e0e0)',
      }}>
        <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 4, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('transactions.filters.source_account')}
            </label>
            <input
              type="text"
              placeholder="G..."
              value={filters.sourceAccount || ''}
              onChange={(e) => handleFilterChange('sourceAccount', e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 4,
                fontSize: 14,
                background: 'var(--bg-primary, #fff)',
                color: 'var(--text-primary, #1a202c)',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('transactions.filters.operation_type')}
            </label>
            <select
              value={filters.operationType || ''}
              onChange={(e) => handleFilterChange('operationType', e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 4,
                fontSize: 14,
                background: 'var(--bg-primary, #fff)',
                color: 'var(--text-primary, #1a202c)',
              }}
            >
              <option value="">{t('transactions.filters.all_types')}</option>
              <option value="payment">{t('transactions.filters.types.payment')}</option>
              <option value="create_account">{t('transactions.filters.types.create_account')}</option>
              <option value="change_trust">{t('transactions.filters.types.change_trust')}</option>
              <option value="path_payment">{t('transactions.filters.types.path_payment')}</option>
              <option value="manage_buy_offer">{t('transactions.filters.types.manage_buy_offer')}</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('transactions.filters.start_date')}
            </label>
            <input
              type="date"
              value={filters.startDate || ''}
              onChange={(e) => handleFilterChange('startDate', e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 4,
                fontSize: 14,
                background: 'var(--bg-primary, #fff)',
                color: 'var(--text-primary, #1a202c)',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 4, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('transactions.filters.end_date')}
            </label>
            <input
              type="date"
              value={filters.endDate || ''}
              onChange={(e) => handleFilterChange('endDate', e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 4,
                fontSize: 14,
                background: 'var(--bg-primary, #fff)',
                color: 'var(--text-primary, #1a202c)',
              }}
            />
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <button
            onClick={() => setFilters({})}
            style={{
              padding: '6px 12px',
              background: 'var(--text-muted, #6c757d)',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            {t('transactions.filters.clear')}
          </button>
        </div>
      </div>

      <TransactionHistoryTable
        response={history}
        loading={loading}
        page={page}
        pageSize={pageSize}
        onPageChange={setPage}
      />

      {history && (
        <div style={{ fontSize: 13, color: 'var(--text-secondary, #666)', textAlign: 'center' }}>
          {t('transactions.showing', { start: Math.min((page + 1) * pageSize, history.total), total: history.total })}
        </div>
      )}
    </div>
  )
}