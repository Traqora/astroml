import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { I18nextProvider } from 'react-i18next'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { initErrorReporting } from './lib/errorReporting'
import { ThemeProvider } from './contexts/ThemeContext'
import i18n from './i18n'
import './styles/skeleton.css'

// Initialise Sentry (no-op when VITE_SENTRY_DSN is absent)
initErrorReporting()

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Surface errors to the nearest ErrorBoundary instead of swallowing them
      throwOnError: true,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* Top-level boundary — catches errors that escape section-level boundaries */}
    <ErrorBoundary boundary="Application">
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <I18nextProvider i18n={i18n}>
            <App />
          </I18nextProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>
)