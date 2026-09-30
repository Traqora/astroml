/**
 * Error reporting module — issue #292.
 *
 * Provides a thin abstraction over Sentry so the rest of the app never
 * imports Sentry directly.  Sentry is loaded lazily via a dynamic import so
 * it doesn't block the initial page render.
 *
 * When VITE_SENTRY_DSN is not set (local dev, CI) the module falls back to
 * console-only logging, so no Sentry account is required to run the app.
 */

import type { BrowserOptions } from '@sentry/react'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ErrorContext {
  /** Human-readable label for the region of the UI that threw. */
  boundary?: string
  /** Additional arbitrary key/value metadata attached to the Sentry event. */
  extra?: Record<string, unknown>
  /** Stack trace string (passed through from ErrorBoundary). */
  componentStack?: string
}

// ── Internal state ────────────────────────────────────────────────────────────

let _initialised = false

// ── Initialisation ────────────────────────────────────────────────────────────

/**
 * Call once at app startup (main.tsx) to configure Sentry.
 * Safe to call when DSN is absent — degrades gracefully.
 */
export async function initErrorReporting(): Promise<void> {
  const dsn = import.meta.env.VITE_SENTRY_DSN as string | undefined
  if (!dsn) return

  try {
    const Sentry = await import('@sentry/react')
    const options: BrowserOptions = {
      dsn,
      environment: import.meta.env.MODE,
      release: import.meta.env.VITE_APP_VERSION as string | undefined,
      // Only send a fraction of traces in production to control volume
      tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
      // Don't capture noise from browser extensions
      denyUrls: [/extensions\//i, /^chrome:\/\//i],
    }
    Sentry.init(options)
    _initialised = true
  } catch {
    // Sentry is an optional dependency — never crash the app if it fails
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Capture an Error (or any thrown value) and forward it to Sentry + the
 * backend logging endpoint.
 */
export function captureError(error: unknown, context: ErrorContext = {}): void {
  const err = toError(error)

  // 1. Always log to console so devs see it even without Sentry
  console.error('[ErrorBoundary]', err, context)

  // 2. Send to Sentry if available
  if (_initialised) {
    import('@sentry/react').then((Sentry) => {
      Sentry.withScope((scope) => {
        if (context.boundary) scope.setTag('boundary', context.boundary)
        if (context.componentStack) scope.setExtra('componentStack', context.componentStack)
        if (context.extra) {
          Object.entries(context.extra).forEach(([k, v]) => scope.setExtra(k, v))
        }
        Sentry.captureException(err)
      })
    }).catch(() => {/* ignore */})
  }

  // 3. Report to our own backend (fire-and-forget, never throws)
  reportToBackend(err, context).catch(() => {/* ignore */})
}

/**
 * Attach user identity to Sentry sessions (call after successful login).
 */
export function setErrorReportingUser(user: { id: string; username?: string } | null): void {
  if (!_initialised) return
  import('@sentry/react').then((Sentry) => {
    Sentry.setUser(user ? { id: user.id, username: user.username } : null)
  }).catch(() => {/* ignore */})
}

// ── Backend error logging ─────────────────────────────────────────────────────

interface BackendErrorPayload {
  message: string
  stack?: string
  boundary?: string
  component_stack?: string
  extra?: Record<string, unknown>
  user_agent: string
  url: string
  timestamp: string
}

async function reportToBackend(err: Error, context: ErrorContext): Promise<void> {
  const apiBase = import.meta.env.VITE_API_BASE_URL || ''
  const payload: BackendErrorPayload = {
    message: err.message,
    stack: err.stack,
    boundary: context.boundary,
    component_stack: context.componentStack,
    extra: context.extra,
    user_agent: navigator.userAgent,
    url: window.location.href,
    timestamp: new Date().toISOString(),
  }

  await fetch(`${apiBase}/api/v1/errors/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // Best-effort — use keepalive so the request survives page unload
    keepalive: true,
    body: JSON.stringify(payload),
  })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function toError(value: unknown): Error {
  if (value instanceof Error) return value
  return new Error(typeof value === 'string' ? value : JSON.stringify(value))
}
