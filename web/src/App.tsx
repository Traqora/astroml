import { lazy, Suspense, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ThemeToggle } from './components/ThemeToggle'
import { useMediaQuery } from './hooks/useMediaQuery'
import {
  SkeletonModelMonitoring,
  SkeletonLoyaltyDashboard,
  SkeletonTransactionHistory,
} from './components/Skeletons'
import { LanguageSwitcher } from './components/i18n'
import './styles/skeleton.css'

// Lazy-load each dashboard section so the initial bundle is smaller and the
// browser can start rendering the first panel before the others are parsed.
const ModelMonitoringDashboard = lazy(() =>
  import('./components/ModelMonitoringDashboard/ModelMonitoringDashboard').then((m) => ({
    default: m.ModelMonitoringDashboard,
  }))
)

const LoyaltyDashboard = lazy(() =>
  import('./components/LoyaltyDashboard').then((m) => ({ default: m.LoyaltyDashboard }))
)

const TransactionHistoryPage = lazy(() =>
  import('./components/TransactionHistory').then((m) => ({ default: m.TransactionHistoryPage }))
)

const sections = [
  { id: 'model-monitoring', label: 'Model Performance' },
  { id: 'loyalty', label: 'Loyalty Dashboard' },
  { id: 'transactions', label: 'Transaction History' },
]

function NavBar() {
  const { t } = useTranslation()
  const isMobile = useMediaQuery('(max-width: 640px)')
  const [menuOpen, setMenuOpen] = useState(false)

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMenuOpen(false)
  }

  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 24,
      paddingBottom: 16,
      borderBottom: '1px solid var(--border-color, #ddd)',
      position: 'relative',
    }}>
      <h1 style={{ margin: 0, fontSize: isMobile ? 18 : 24, fontWeight: 700 }}>
        {t('app.title')}
      </h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {isMobile ? (
          <>
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              aria-label="Toggle navigation menu"
              style={{
                background: 'none',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 6,
                padding: '6px 10px',
                cursor: 'pointer',
                color: 'var(--text-primary, #1a202c)',
                fontSize: 18,
              }}
            >
              {menuOpen ? '✕' : '☰'}
            </button>
            <ThemeToggle />
            <LanguageSwitcher />
            {menuOpen && (
              <div style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                left: 0,
                background: 'var(--bg-card, #fff)',
                border: '1px solid var(--border-color, #ddd)',
                borderRadius: 8,
                padding: 8,
                zIndex: 100,
                boxShadow: 'var(--shadow-md, 0 2px 14px rgba(0,0,0,0.1))',
              }}>
                {sections.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => scrollTo(s.id)}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '10px 12px',
                      background: 'none',
                      border: 'none',
                      textAlign: 'left',
                      cursor: 'pointer',
                      color: 'var(--text-primary, #1a202c)',
                      fontSize: 14,
                      fontWeight: 600,
                      borderRadius: 4,
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <div style={{ display: 'flex', gap: 8 }}>
              {sections.map((s) => (
                <button
                  key={s.id}
                  onClick={() => scrollTo(s.id)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: 6,
                    border: '1px solid var(--border-color, #ddd)',
                    background: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-primary, #1a202c)',
                    fontSize: 13,
                    fontWeight: 600,
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <ThemeToggle />
            <LanguageSwitcher />
          </>
        )}
      </div>
    </nav>
  )
}

export default function App() {
  const { t } = useTranslation()
  const isMobile = useMediaQuery('(max-width: 640px)')

  return (
    <div style={{
      fontFamily: 'system-ui, sans-serif',
      padding: isMobile ? 12 : 16,
      maxWidth: 1200,
      margin: '0 auto',
    }}>
      <NavBar />

      <h1 id="model-monitoring">{t('app.title')}</h1>
      <ErrorBoundary boundary="Model Monitoring">
        <Suspense fallback={<SkeletonModelMonitoring />}>
          <ModelMonitoringDashboard />
        </Suspense>
      </ErrorBoundary>

      <hr style={{ margin: isMobile ? '24px 0' : '40px 0', borderColor: 'var(--border-color, #ddd)' }} />

      <h1 id="loyalty">{t('app.loyalty')}</h1>
      <ErrorBoundary boundary="Loyalty Dashboard">
        <Suspense fallback={<SkeletonLoyaltyDashboard />}>
          <LoyaltyDashboard />
        </Suspense>
      </ErrorBoundary>

      <hr style={{ margin: isMobile ? '24px 0' : '40px 0', borderColor: 'var(--border-color, #ddd)' }} />

      <h1 id="transactions">{t('app.transactions')}</h1>
      <ErrorBoundary boundary="Transaction History">
        <Suspense fallback={<SkeletonTransactionHistory />}>
          <TransactionHistoryPage />
        </Suspense>
      </ErrorBoundary>
    </div>
  )
}