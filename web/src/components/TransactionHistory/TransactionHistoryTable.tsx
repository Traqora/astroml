import React from 'react'
import { useTranslation } from 'react-i18next'
import type { BlockchainTransaction, TransactionHistoryResponse } from '../../lib/types'

export function TransactionHistoryTable({
  response,
  loading,
  page,
  pageSize,
  onPageChange,
}: {
  response: TransactionHistoryResponse | undefined
  loading: boolean
  page: number
  pageSize: number
  onPageChange: (p: number) => void
}) {
  const { t } = useTranslation()
  const total = response?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  const formatHash = (hash: string) => {
    return `${hash.slice(0, 8)}...${hash.slice(-8)}`
  }

  const formatAddress = (address: string) => {
    return `${address.slice(0, 4)}...${address.slice(-4)}`
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: '8px 0' }}>{t('transactions.table.title')}</h2>
        <div>
          <button 
            disabled={page === 0 || loading} 
            onClick={() => onPageChange(page - 1)}
            style={{ padding: '6px 12px', marginRight: '8px', cursor: page === 0 || loading ? 'not-allowed' : 'pointer' }}
          >
            {t('transactions.table.prev')}
          </button>
          <span style={{ margin: '0 8px' }}>{t('transactions.table.page', { current: page + 1, total: totalPages })}</span>
          <button 
            disabled={page + 1 >= totalPages || loading} 
            onClick={() => onPageChange(page + 1)}
            style={{ padding: '6px 12px', marginLeft: '8px', cursor: page + 1 >= totalPages || loading ? 'not-allowed' : 'pointer' }}
          >
            {t('transactions.table.next')}
          </button>
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '16px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={th}>{t('transactions.table.columns.hash')}</th>
              <th style={th}>{t('transactions.table.columns.ledger')}</th>
              <th style={th}>{t('transactions.table.columns.source')}</th>
              <th style={th}>{t('transactions.table.columns.destination')}</th>
              <th style={th}>{t('transactions.table.columns.type')}</th>
              <th style={th}>{t('transactions.table.columns.amount')}</th>
              <th style={th}>{t('transactions.table.columns.asset')}</th>
              <th style={th}>{t('transactions.table.columns.fee')}</th>
              <th style={th}>{t('transactions.table.columns.status')}</th>
              <th style={th}>{t('transactions.table.columns.date')}</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={10} style={{ padding: 12, textAlign: 'center' }}>{t('transactions.table.loading')}</td></tr>
            )}
            {!loading && response?.data.length === 0 && (
              <tr><td colSpan={10} style={{ padding: 12, textAlign: 'center' }}>{t('transactions.table.no_data')}</td></tr>
            )}
            {!loading && response?.data.map((tx) => (
              <tr key={tx.hash}>
                <td style={td}>
                  <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                    {formatHash(tx.hash)}
                  </span>
                </td>
                <td style={td}>{tx.ledgerSequence}</td>
                <td style={td}>
                  <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                    {formatAddress(tx.sourceAccount)}
                  </span>
                </td>
                <td style={td}>
                  {tx.destinationAccount ? (
                    <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                      {formatAddress(tx.destinationAccount)}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted, #999)' }}>{t('transactions.table.placeholder')}</span>
                  )}
                </td>
                <td style={td}>{tx.operationType}</td>
                <td style={td}>
                  {tx.amount !== undefined ? tx.amount.toLocaleString() : '-'}
                </td>
                <td style={td}>{tx.assetCode || 'XLM'}</td>
                <td style={td}>{tx.fee} {t('transactions.table.fee_unit')}</td>
                <td style={td}>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    backgroundColor: tx.successful ? '#d4edda' : '#f8d7da',
                    color: tx.successful ? '#155724' : '#721c24',
                    border: '1px solid transparent',
                  }}>
                    {tx.successful ? t('transactions.table.success') : t('transactions.table.failed')}
                  </span>
                </td>
                <td style={td}>
                  {new Date(tx.createdAt).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const th: React.CSSProperties = { 
  textAlign: 'left', 
  borderBottom: '2px solid var(--border-color, #ddd)', 
  padding: 12,
  fontWeight: 600,
  fontSize: '13px',
  color: 'var(--text-secondary, #555)'
}
const td: React.CSSProperties = { 
  borderBottom: '1px solid var(--border-light, #f1f1f1)', 
  padding: 10,
  fontSize: '13px'
}