import { useTheme } from '../contexts/ThemeContext'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
      style={{
        padding: '6px 14px',
        borderRadius: 6,
        border: '1px solid var(--border-color, #ddd)',
        background: 'var(--bg-card, #fff)',
        color: 'var(--text-primary, #1a202c)',
        cursor: 'pointer',
        fontSize: 14,
        fontWeight: 600,
      }}
    >
      {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
    </button>
  )
}
