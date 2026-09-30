import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAccountDetail, useAccountFraudSummary, useAccountLoyaltySummary, useAccountTransactions } from '../../hooks/useAccountDetail'
import { SkeletonCard } from '../Skeletons'

export function AccountDetailPage({ publicKey }: { publicKey: string }) {
  const { t } = useTranslation()
  const [page, setPage] = useState(0)
  const pageSize = 10

  const { data: account, isLoading: loadingAccount } = useAccountDetail(publicKey)
  const { data: fraud } = useAccountFraudSummary(publicKey)
  const { data: loyalty } = useAccountLoyaltySummary(publicKey)
  const { data: transactions, isLoading: loadingTx } = useAccountTransactions(publicKey, page, pageSize)

  if (loadingAccount) return <SkeletonCard />

  if (!account) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted, #666)' }}>
        {t('account.not_found')}
      </div>
    )
  }

  const totalPages = Math.max(1, Math.ceil((transactions?.total ?? 0) / pageSize))

  return (
    <div style={{ display: 'grid', gap: 24 }}>
      <div>
        <h2 style={{ margin: '0 0 8px 0', fontSize: 22, fontWeight: 700 }}>{t('account.title')}</h2>
        <p style={{ margin: 0, color: 'var(--text-muted, #666)', fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>
          {publicKey}
        </p>
      </div>

      <div style={{
        display: 'grid',
        gap: 16,
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      }}>
        <InfoCard label={t('account.fields.balance')} value={account.balance != null ? `${account.balance.toLocaleString()} XLM` : '-'} />
        <InfoCard label={t('account.fields.sequence')} value={account.sequence != null ? String(account.sequence) : '-'} />
        <InfoCard label={t('account.fields.home_domain')} value={account.homeDomain || '-'} />
        <InfoCard label={t('account.fields.flags')} value={String(account.flags)} />
      </div>

      <div style={{
        display: 'grid',
        gap: 16,
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      }}>
        {fraud && (
          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: 14, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('account.fraud.title')}
            </h3>
            <div style={{ display: 'grid', gap: 4, fontSize: 13 }}>
              <span>{t('account.fraud.total')}: <strong>{fraud.totalAlerts}</strong></span>
              <span style={{ color: '#dc3545' }}>{t('account.fraud.high')}: {fraud.highRisk}</span>
              <span style={{ color: '#ffc107' }}>{t('account.fraud.medium')}: {fraud.mediumRisk}</span>
              <span style={{ color: '#28a745' }}>{t('account.fraud.low')}: {fraud.lowRisk}</span>
            </div>
          </div>
        )}

        {loyalty && (
          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: 14, fontWeight: 600, color: 'var(--text-secondary, #555)' }}>
              {t('account.loyalty.title')}
            </h3>
            <div style={{ display: 'grid', gap: 4, fontSize: 13 }}>
              <span>{t('account.loyalty.tier')}: <strong>{loyalty.tierName}</strong></span>
              <span>{t('account.loyalty.points')}: <strong>{loyalty.pointsBalance.toLocaleString()}</strong></span>
            </div>
          </div>
        )}
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{t('account.transactions.title')}</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              disabled={page === 0 || loadingTx}
              onClick={() => setPage((p) => p - 1)}
              style={paginationBtnStyle(page === 0 || loadingTx)}
            >
              {t('transactions.table.prev')}
            </button>
            <span style={{ fontSize: 13, color: 'var(--text-secondary, #666)' }}>
              {t('transactions.table.page', { current: page + 1, total: totalPages })}
            </span>
            <button
              disabled={page + 1 >= totalPages || loadingTx}
              onClick={() => setPage((p) => p + 1)}
              style={paginationBtnStyle(page + 1 >= totalPages || loadingTx)}
            >
              {t('transactions.table.next')}
            </button>
          </div>
        </div>

        {loadingTx ? (
          <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted, #666)' }}>{t('common.loading')}</div>
        ) : transactions?.data.length === 0 ? (
          <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted, #666)' }}>{t('account.transactions.none')}</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={th}>{t('transactions.table.columns.hash')}</th>
                  <th style={th}>{t('transactions.table.columns.ledger')}</th>
                  <th style={th}>{t('transactions.table.columns.source')}</th>
                  <th style={th}>{t('transactions.table.columns.destination')}</th>
                  <th style={th}>{t('transactions.table.columns.type')}</th>
                  <th style={th}>{t('transactions.table.columns.amount')}</th>
                  <th style={th}>{t('transactions.table.columns.fee')}</th>
                  <th style={th}>{t('transactions.table.columns.status')}</th>
                  <th style={th}>{t('transactions.table.columns.date')}</th>
                </tr>
              </thead>
              <tbody>
                {transactions?.data.map((tx) => (
                  <tr key={tx.hash}>
                    <td style={td}><span style={{ fontFamily: 'monospace', fontSize: 12 }}>{tx.hash.slice(0, 8)}...{tx.hash.slice(-8)}</span></td>
                    <td style={td}>{tx.ledgerSequence}</td>
                    <td style={td}><span style={{ fontFamily: 'monospace', fontSize: 12 }}>{tx.sourceAccount.slice(0, 4)}...{tx.sourceAccount.slice(-4)}</span></td>
                    <td style={td}>
                      {tx.destinationAccount ? (
                        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{tx.destinationAccount.slice(0, 4)}...{tx.destinationAccount.slice(-4)}</span>
                      ) : (
                        <span style={{ color: 'var(--text-muted, #999)' }}>-</span>
                      )}
                    </td>
                    <td style={td}>{tx.operationType || '-'}</td>
                    <td style={td}>{tx.amount != null ? tx.amount.toLocaleString() : '-'}</td>
                    <td style={td}>{tx.fee}</td>
                    <td style={td}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: 12,
                        backgroundColor: tx.successful ? '#d4edda' : '#f8d7da',
                        color: tx.successful ? '#155724' : '#721c24',
                      }}>
                        {tx.successful ? t('transactions.table.success') : t('transactions.table.failed')}
                      </span>
                    </td>
                    <td style={td}>{new Date(tx.createdAt).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary, #555)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 500 }}>{value}</div>
    </div>
  )
}

const cardStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--border-color, #e0e0e0)',
  backgroundColor: 'var(--bg-card, #fff)',
}

const th: React.CSSProperties = {
  textAlign: 'left',
  borderBottom: '2px solid var(--border-color, #ddd)',
  padding: 10,
  fontWeight: 600,
  fontSize: 13,
  color: 'var(--text-secondary, #555)',
}

const td: React.CSSProperties = {
  borderBottom: '1px solid var(--border-light, #f1f1f1)',
  padding: 8,
  fontSize: 13,
}

function paginationBtnStyle(disabled: boolean): React.CSSProperties {
  return {
    padding: '6px 12px',
    border: '1px solid var(--border-color, #ddd)',
    borderRadius: 4,
    background: 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    fontSize: 13,
    color: 'var(--text-primary, #1a202c)',
  }
}
