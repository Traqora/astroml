import { memo } from 'react'
import { useTranslation } from 'react-i18next'

interface LastUpdatedProps {
  timestamp: Date | null
  isRealtime?: boolean
}

export const LastUpdated = memo(function LastUpdated({
  timestamp,
  isRealtime = false,
}: LastUpdatedProps) {
  const { t } = useTranslation()

  if (!timestamp) return null

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#888' }}>
      {isRealtime && (
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: '#22c55e',
            animation: 'pulse 2s infinite',
          }}
        />
      )}
      <span>
        {t('monitoring.last_updated')}: {timestamp.toLocaleTimeString()}
      </span>
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
          }
        `}
      </style>
    </div>
  )
})