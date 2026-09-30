import { lazy, Suspense, useState, useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ThemeToggle } from './components/ThemeToggle'
import { useMediaQuery } from './hooks/useMediaQuery'
import {
  SkeletonModelMonitoring,
  SkeletonLoyaltyDashboard,
  SkeletonTransactionHistory,
  SkeletonCard,
} from './components/Skeletons'
import { LanguageSwitcher } from './components/i18n'
import { AccountSearchBar } from './components/AccountSearchBar'
import './styles/skeleton.css'

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

const AccountDetailPage = lazy(() =>
  import('./components/AccountDetailPage').then((m) => ({ default: m.AccountDetailPage }))
)

const TransactionDetailPage = lazy(() =>
  import('./components/TransactionDetailPage').then((m) => ({ default: m.TransactionDetailPage }))
)

type View =
  | { kind: 'home' }
  | { kind: 'account'; publicKey: string }
  | { kind: 'transaction'; hash: string }

const sections = [
  { id: 'model-monitoring', label: 'Model Performance' },
  { id: 'loyalty', label: 'Loyalty Dashboard' },
  { id: 'transactions', label: 'Transaction History' },
]

const MOBILE_MENU_ID = 'mobile-nav-menu'

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export function NavBar({ view, onNavigate }: { view: View; onNavigate: (v: View) => void }) {
  const { t } = useTranslation()
  const isMobile = useMediaQuery('(max-width: 640px)')
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const menuToggleRef = useRef<HTMLButtonElement>(null)

  const closeMenu = useCallback(() => setMenuOpen(false), [])

  // Keyboard flow: Escape closes the mobile menu and returns focus to its
  // trigger, instead of stranding focus on a now-hidden element.
  useEffect(() => {
    if (!menuOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeMenu()
        menuToggleRef.current?.focus()
      }
    }
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (!menuRef.current?.contains(target) && !menuToggleRef.current?.contains(target)) {
        closeMenu()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('mousedown', handlePointerDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('mousedown', handlePointerDown)
    }
  }, [menuOpen, closeMenu])

  const scrollTo = (id: string) => {
    const behavior = prefersReducedMotion() ? 'auto' : 'smooth'
    if (view.kind !== 'home') {
      onNavigate({ kind: 'home' })
      requestAnimationFrame(() => {
        document.getElementById(id)?.scrollIntoView({ behavior })
      })
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior })
    }
    closeMenu()
  }

  return (
    <nav aria-label="Main" style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 24,
      paddingBottom: 16,
      borderBottom: '1px solid var(--border-color, #ddd)',
      position: 'relative',
    }}>
      <button
        onClick={() => onNavigate({ kind: 'home' })}
        style={{ margin: 0, fontSize: isMobile ? 18 : 24, fontWeight: 700, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary, #1a202c)' }}
      >
        {t('app.title')}
      </button>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {isMobile ? (
          <>
            <button
              ref={menuToggleRef}
              onClick={() => setMenuOpen((open) => !open)}
              aria-label="Toggle navigation menu"
              aria-haspopup="true"
              aria-expanded={menuOpen}
              aria-controls={MOBILE_MENU_ID}
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
              <div
                id={MOBILE_MENU_ID}
                ref={menuRef}
                style={{
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
                }}
              >
                <nav aria-label="Section links">
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
                </nav>
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
  const [view, setView] = useState<View>({ kind: 'home' })

  const navigateToAccount = useCallback((publicKey: string) => {
    setView({ kind: 'account', publicKey })
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
  }, [])

  const navigateToTransaction = useCallback((hash: string) => {
    setView({ kind: 'transaction', hash })
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
  }, [])

  const handleSearchSelect = useCallback((result: { id: string; type: string }) => {
    if (result.type === 'account') {
      navigateToAccount(result.id)
    } else {
      navigateToTransaction(result.id)
    }
  }, [navigateToAccount, navigateToTransaction])

  return (
    <div style={{
      fontFamily: 'system-ui, sans-serif',
      padding: isMobile ? 12 : 16,
      maxWidth: 1200,
      margin: '0 auto',
    }}>
      <a href="#main-content" className="skip-link">
        {t('app.skipToContent', 'Skip to main content')}
      </a>

      <NavBar view={view} onNavigate={setView} />

      <div style={{ marginBottom: 24, maxWidth: 480 }}>
        <AccountSearchBar onSelectResult={handleSearchSelect} />
      </div>

      {/* tabIndex=-1: not in the tab order itself, but focusable as the
          skip link's target so keyboard/AT users actually land here. */}
      <main id="main-content" tabIndex={-1}>
        {view.kind === 'account' && (
          <Suspense fallback={<SkeletonCard />}>
            <AccountDetailPage publicKey={view.publicKey} />
          </Suspense>
        )}

        {view.kind === 'transaction' && (
          <Suspense fallback={<SkeletonCard />}>
            <TransactionDetailPage hash={view.hash} />
          </Suspense>
        )}

        {view.kind === 'home' && (
          <>
            <h1 id="model-monitoring">{t('app.title')}</h1>
            <ErrorBoundary boundary="Model Monitoring">
              <Suspense fallback={<SkeletonModelMonitoring />}>
                <ModelMonitoringDashboard />
              </Suspense>
            </ErrorBoundary>

            <hr style={{ margin: isMobile ? '24px 0' : '40px 0', borderColor: 'var(--border-color, #ddd)' }} />

            <h2 id="loyalty">{t('app.loyalty')}</h2>
            <ErrorBoundary boundary="Loyalty Dashboard">
              <Suspense fallback={<SkeletonLoyaltyDashboard />}>
                <LoyaltyDashboard />
              </Suspense>
            </ErrorBoundary>

            <hr style={{ margin: isMobile ? '24px 0' : '40px 0', borderColor: 'var(--border-color, #ddd)' }} />

            <h2 id="transactions">{t('app.transactions')}</h2>
            <ErrorBoundary boundary="Transaction History">
              <Suspense fallback={<SkeletonTransactionHistory />}>
                <TransactionHistoryPage onAccountClick={navigateToAccount} onTransactionClick={navigateToTransaction} />
              </Suspense>
            </ErrorBoundary>
          </>
        )}
      </main>
    </div>
  )
}
