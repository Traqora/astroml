/**
 * ErrorBoundary component — issue #292.
 *
 * A React class component (required because only class components can
 * implement componentDidCatch / getDerivedStateFromError).
 *
 * Features:
 *  - Catches any render/lifecycle error in the subtree
 *  - Reports to Sentry + backend via captureError()
 *  - Shows a user-friendly UI with a retry button and a "report" link
 *  - Accepts an optional `fallback` prop for custom error UIs
 *  - Accepts an optional `boundary` prop (label used in error reports)
 *  - Reset on `resetKeys` prop change (mirrors react-error-boundary convention)
 */

import { Component, type ErrorInfo, type ReactNode } from 'react'
import { captureError } from '../../lib/errorReporting'

// ── Props & State ─────────────────────────────────────────────────────────────

export interface ErrorBoundaryProps {
  children: ReactNode
  /**
   * Human-readable label for this boundary region — shown in error reports
   * and used as a Sentry tag so errors are grouped by feature area.
   */
  boundary?: string
  /**
   * Custom fallback UI.  Receives the error and a `reset` callback so the
   * fallback can offer an inline "try again" action.
   */
  fallback?: (error: Error, reset: () => void) => ReactNode
  /**
   * When any of these values change the boundary is automatically reset.
   * Useful when wrapping route-level components that change on navigation.
   */
  resetKeys?: unknown[]
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

// ── Component ─────────────────────────────────────────────────────────────────

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null }

  // React calls this synchronously to update state before the next render
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  // React calls this after the error has been committed to the DOM
  componentDidCatch(error: Error, info: ErrorInfo): void {
    captureError(error, {
      boundary: this.props.boundary,
      componentStack: info.componentStack ?? undefined,
    })
  }

  // Watch resetKeys — if they change while in an error state, reset
  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    const { resetKeys } = this.props
    if (this.state.hasError && resetKeys) {
      const changed = resetKeys.some((key, i) => key !== prevProps.resetKeys?.[i])
      if (changed) this._reset()
    }
  }

  private _reset = (): void => {
    this.setState({ hasError: false, error: null })
  }

  render(): ReactNode {
    const { hasError, error } = this.state
    const { children, fallback, boundary } = this.props

    if (!hasError || !error) return children

    // Use custom fallback if provided
    if (fallback) return fallback(error, this._reset)

    // Default built-in error UI
    return <DefaultErrorUI error={error} boundary={boundary} onReset={this._reset} />
  }
}

// ── Default error UI ──────────────────────────────────────────────────────────

interface DefaultErrorUIProps {
  error: Error
  boundary?: string
  onReset: () => void
}

function DefaultErrorUI({ error, boundary, onReset }: DefaultErrorUIProps) {
  const title = boundary ? `Something went wrong in ${boundary}` : 'Something went wrong'

  return (
    <div role="alert" aria-live="assertive" style={containerStyle}>
      <div style={iconStyle} aria-hidden="true">⚠️</div>

      <h2 style={headingStyle}>{title}</h2>

      <p style={bodyStyle}>
        An unexpected error occurred. The issue has been reported automatically.
        You can try reloading this section or refresh the page.
      </p>

      <details style={detailsStyle}>
        <summary style={summaryStyle}>Error details</summary>
        <pre style={preStyle}>{error.message}</pre>
      </details>

      <div style={actionsStyle}>
        <button
          type="button"
          onClick={onReset}
          style={primaryButtonStyle}
          aria-label="Retry loading this section"
        >
          Try again
        </button>
        <button
          type="button"
          onClick={() => window.location.reload()}
          style={secondaryButtonStyle}
          aria-label="Reload the full page"
        >
          Reload page
        </button>
      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '40px 24px',
  margin: '16px 0',
  borderRadius: 12,
  background: 'var(--bg-card, #fff8f8)',
  border: '1px solid var(--border-color, #fed7d7)',
  textAlign: 'center',
  maxWidth: 520,
  marginLeft: 'auto',
  marginRight: 'auto',
}

const iconStyle: React.CSSProperties = {
  fontSize: 40,
  marginBottom: 12,
}

const headingStyle: React.CSSProperties = {
  margin: '0 0 8px',
  fontSize: 18,
  fontWeight: 700,
  color: 'var(--text-primary, #c53030)',
}

const bodyStyle: React.CSSProperties = {
  margin: '0 0 16px',
  fontSize: 14,
  color: 'var(--text-secondary, #4a5568)',
  lineHeight: 1.6,
}

const detailsStyle: React.CSSProperties = {
  width: '100%',
  textAlign: 'left',
  marginBottom: 20,
}

const summaryStyle: React.CSSProperties = {
  cursor: 'pointer',
  fontSize: 12,
  color: 'var(--text-muted, #718096)',
  userSelect: 'none',
}

const preStyle: React.CSSProperties = {
  marginTop: 8,
  padding: '8px 12px',
  background: 'var(--bg-secondary, #fff5f5)',
  border: '1px solid var(--border-color, #fed7d7)',
  borderRadius: 6,
  fontSize: 11,
  overflowX: 'auto',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  color: 'var(--text-primary, #c53030)',
}

const actionsStyle: React.CSSProperties = {
  display: 'flex',
  gap: 10,
  flexWrap: 'wrap',
  justifyContent: 'center',
}

const baseButtonStyle: React.CSSProperties = {
  padding: '8px 20px',
  borderRadius: 6,
  fontSize: 13,
  fontWeight: 600,
  cursor: 'pointer',
  border: 'none',
}

const primaryButtonStyle: React.CSSProperties = {
  ...baseButtonStyle,
  background: '#e53e3e',
  color: '#fff',
}

const secondaryButtonStyle: React.CSSProperties = {
  ...baseButtonStyle,
  background: 'var(--bg-secondary, #edf2f7)',
  color: 'var(--text-primary, #2d3748)',
}
