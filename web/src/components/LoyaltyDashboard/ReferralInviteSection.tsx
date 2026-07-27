import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { getReferralLink } from '../../api/loyalty'

export function ReferralInviteSection() {
  const { t } = useTranslation()
  const { data } = useQuery({ queryKey: ['referral'], queryFn: getReferralLink })
  const url = data?.url ?? ''

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      alert(t('loyalty.referral.copied'))
    } catch {
      // noop
    }
  }

  const share = async () => {
    if ((navigator as any).share) {
      try {
        await (navigator as any).share({ title: t('loyalty.referral.share_title'), text: t('loyalty.referral.share_text'), url })
      } catch {
        // ignore
      }
    } else {
      copy()
    }
  }

  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>{t('loyalty.referral.title')}</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <input style={{ minWidth: 260 }} value={url} readOnly />
        <button onClick={copy}>{t('loyalty.referral.copy')}</button>
        <button onClick={share}>{t('loyalty.referral.share')}</button>
        <div style={{ color: 'var(--text-secondary, #555)' }}>
          {t('loyalty.referral.stats', { invited: data?.invited ?? 0, rewards: data?.rewards ?? 0 })}
        </div>
      </div>
    </div>
  )
}