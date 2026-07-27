import { useState } from 'react'
import { useTranslation } from 'react-i18next'

export function PointsRedemptionPanel({ balance, onRedeem, pending }: { balance: number; onRedeem: (points: number) => void; pending: boolean }) {
  const { t } = useTranslation()
  const [points, setPoints] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const submit = () => {
    setError(null)
    if (points <= 0) {
      setError(t('loyalty.redeem.errors.positive'))
      return
    }
    if (points > balance) {
      setError(t('loyalty.redeem.errors.exceeds_balance'))
      return
    }
    onRedeem(points)
    setPoints(0)
  }

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>{t('loyalty.redeem.title')}</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          type="number"
          min={1}
          value={points}
          onChange={(e) => setPoints(parseInt(e.target.value, 10) || 0)}
          disabled={pending}
        />
        <button onClick={submit} disabled={pending}>{t('loyalty.redeem.button')}</button>
        <div style={{ color: 'var(--text-secondary, #555)' }}>{t('loyalty.redeem.available', { balance: balance.toLocaleString() })}</div>
      </div>
      {error && <div style={{ color: '#e53e3e', marginTop: 8 }}>{error}</div>}
    </div>
  )
}