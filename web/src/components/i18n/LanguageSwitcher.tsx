import { useTranslation } from 'react-i18next'

const languages = [
  { code: 'en', label: 'English', flag: '🇬🇧' },
  { code: 'es', label: 'Español', flag: '🇪🇸' },
  { code: 'zh', label: '中文', flag: '🇨🇳' },
  { code: 'ja', label: '日本語', flag: '🇯🇵' },
]

export function LanguageSwitcher() {
  const { i18n } = useTranslation()
  const currentLang = i18n.language

  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      {languages.map(({ code, label, flag }) => (
        <button
          key={code}
          onClick={() => changeLanguage(code)}
          style={{
            padding: '4px 10px',
            border: currentLang === code ? '2px solid #3b82f6' : '1px solid #d1d5db',
            borderRadius: 6,
            background: currentLang === code ? '#eff6ff' : 'transparent',
            cursor: 'pointer',
            fontSize: 14,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            transition: 'all 0.2s',
          }}
          aria-label={`Switch to ${label}`}
          title={`Switch to ${label}`}
        >
          <span>{flag}</span>
          <span style={{ fontSize: 12 }}>{label}</span>
        </button>
      ))}
    </div>
  )
}