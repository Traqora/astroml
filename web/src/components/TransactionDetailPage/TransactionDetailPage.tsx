import { useTranslation } from 'react-i18next'
import { useTransactionDetail } from '../../hooks/useTransactionDetail'
import { SkeletonCard } from '../Skeletons'

export function TransactionDetailPage({ hash }: { hash: string }) {
  const { t } = useTranslation()
  const { data: tx, isLoading } = useTransactionDetail(hash)

  if (isLoading) return <SkeletonCard />

  if (!tx) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted, #666)' }}>
        {t('transaction.not_found')}
      </div>
    )
  }

  return (
    <div style={{ display: 'grid', gap: 24 }}>
      <div>
        <h2 style={{ margin: '0 0 8px 0', fontSize: 22, fontWeight: 700 }}>{t('transaction.title')}</h2>
        <p style={{ margin: 0, color: 'var(--text-muted, #666)', fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>
          {hash}
        </p>
      </div>

      <div style={{
        display: 'grid',
        gap: 16,
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
      }}>
        <InfoCard label={t('transaction.fields.status')} value={tx.successful ? t('transactions.table.success') : t('transactions.table.failed')} accent={tx.successful ? '#28a745' : '#dc3545'} />
        <InfoCard label={t('transaction.fields.type')} value={tx.operationType || '-'} />
        <InfoCard label={t('transaction.fields.amount')} value={tx.amount != null ? `${tx.amount.toLocaleString()} ${tx.assetCode || 'XLM'}` : '-'} />
        <InfoCard label={t('transaction.fields.fee')} value={`${tx.fee} stroops`} />
        <InfoCard label={t('transaction.fields.ledger')} value={String(tx.ledgerSequence)} />
        <InfoCard label={t('transaction.fields.date')} value={new Date(tx.createdAt).toLocaleString()} />
      </div>

      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
        <div style={cardStyle}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary, #555)', marginBottom: 8 }}>
            {t('transaction.fields.source_account')}
          </div>
          <p style={{ margin: 0, fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>{tx.sourceAccount}</p>
        </div>

        {tx.destinationAccount && (
          <div style={cardStyle}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary, #555)', marginBottom: 8 }}>
              {t('transaction.fields.destination_account')}
            </div>
            <p style={{ margin: 0, fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>{tx.destinationAccount}</p>
          </div>
        )}
      </div>

      {tx.memoType && (
        <div style={cardStyle}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary, #555)', marginBottom: 4 }}>
            {t('transaction.fields.memo_type')}
          </div>
          <span style={{ fontSize: 13 }}>{tx.memoType}</span>
        </div>
      )}
    </div>
  )
}

function InfoCard({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary, #555)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 500, color: accent || 'var(--text-primary, #1a202c)' }}>{value}</div>
    </div>
  )
}

const cardStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--border-color, #e0e0e0)',
  backgroundColor: 'var(--bg-card, #fff)',
}
