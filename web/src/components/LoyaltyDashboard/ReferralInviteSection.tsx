import { useQuery } from '@tanstack/react-query'
import { getReferralLink } from '../../api/loyalty'

export function ReferralInviteSection() {
  const { data } = useQuery({ queryKey: ['referral'], queryFn: getReferralLink })
  const url = data?.url ?? ''

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      alert('Referral link copied!')
    } catch {
      // noop
    }
  }

  const share = async () => {
    if ((navigator as any).share) {
      try {
        await (navigator as any).share({ title: 'Join me!', text: 'Use my referral link', url })
      } catch {
        // ignore
      }
    } else {
      copy()
    }
  }

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>Invite Friends</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <input style={{ minWidth: 260 }} value={url} readOnly />
        <button onClick={copy}>Copy</button>
        <button onClick={share}>Share</button>
        <div style={{ color: '#555' }}>
          Invited: {data?.invited ?? 0} • Rewards: {data?.rewards ?? 0}
        </div>
      </div>
    </div>
  )
}
