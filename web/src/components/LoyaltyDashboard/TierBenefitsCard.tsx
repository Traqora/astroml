import type { Benefit } from '../../lib/types'

export function TierBenefitsCard({ benefits }: { benefits: Benefit[] }) {
  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>Tier Benefits</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
        {benefits.map((b) => (
          <div key={b.id} style={card}>
            <div style={{ fontWeight: 600 }}>{b.title}</div>
            <div style={{ color: '#555', fontSize: 14 }}>{b.description}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

const card: React.CSSProperties = {
  border: '1px solid #eee',
  borderRadius: 8,
  padding: 12,
  background: '#fff',
  boxShadow: '0 1px 2px rgba(0,0,0,0.03)'
}
