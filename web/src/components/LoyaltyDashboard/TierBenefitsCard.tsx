import { useTranslation } from 'react-i18next'
import type { Benefit } from '../../lib/types'

export function TierBenefitsCard({ benefits }: { benefits: Benefit[] }) {
  const { t } = useTranslation()

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>{t('loyalty.benefits')}</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
        {benefits.map((b) => (
          <div key={b.id} style={card}>
            <div style={{ fontWeight: 600 }}>{b.title}</div>
            <div style={{ color: 'var(--text-secondary, #555)', fontSize: 14 }}>{b.description}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

const card: React.CSSProperties = {
  border: '1px solid var(--border-light, #eee)',
  borderRadius: 8,
  padding: 12,
  background: 'var(--bg-card, #fff)',
  boxShadow: '0 1px 2px rgba(0,0,0,0.03)'
}