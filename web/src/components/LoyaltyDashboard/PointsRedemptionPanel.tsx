import { useState } from 'react'

export function PointsRedemptionPanel({ balance, onRedeem, pending }: { balance: number; onRedeem: (points: number) => void; pending: boolean }) {
  const [points, setPoints] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const submit = () => {
    setError(null)
    if (points <= 0) {
      setError('Enter a positive amount')
      return
    }
    if (points > balance) {
      setError('Amount exceeds balance')
      return
    }
    onRedeem(points)
    setPoints(0)
  }

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>Redeem Points</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          type="number"
          min={1}
          value={points}
          onChange={(e) => setPoints(parseInt(e.target.value, 10) || 0)}
          disabled={pending}
        />
        <button onClick={submit} disabled={pending}>Redeem</button>
        <div style={{ color: '#555' }}>Available: {balance.toLocaleString()}</div>
      </div>
      {error && <div style={{ color: 'crimson', marginTop: 8 }}>{error}</div>}
    </div>
  )
}
