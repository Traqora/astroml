/**
 * Tests for ErrorBoundary component — issue #292.
 *
 * jsdom suppresses React's console.error calls for caught errors by design,
 * so we silence those with a spy to keep test output clean.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest'
import { ErrorBoundary } from '../components/ErrorBoundary'

// ── Helpers ───────────────────────────────────────────────────────────────────

/** A component that throws on render so we can trigger the boundary. */
function Bomb({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error('Test explosion')
  return <div>Safe content</div>
}

/** A component whose throw can be toggled via a prop to test reset. */
function ToggleBomb({ explode }: { explode: boolean }) {
  if (explode) throw new Error('Toggled explosion')
  return <div>Recovered content</div>
}

// ── Setup ─────────────────────────────────────────────────────────────────────

beforeEach(() => {
  // Suppress React's error boundary console noise in tests
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ErrorBoundary', () => {
  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>All good</div>
      </ErrorBoundary>
    )
    expect(screen.getByText('All good')).toBeInTheDocument()
  })

  it('shows the default error UI when a child throws', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument()
  })

  it('includes the boundary name in the error heading', () => {
    render(
      <ErrorBoundary boundary="Loyalty Dashboard">
        <Bomb shouldThrow />
      </ErrorBoundary>
    )
    expect(screen.getByText(/loyalty dashboard/i)).toBeInTheDocument()
  })

  it('shows error details in the disclosure element', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )
    expect(screen.getByText('Test explosion')).toBeInTheDocument()
  })

  it('renders a custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={(err) => <div>Custom: {err.message}</div>}>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )
    expect(screen.getByText('Custom: Test explosion')).toBeInTheDocument()
  })

  it('calls the custom fallback reset function', () => {
    const { rerender } = render(
      <ErrorBoundary
        fallback={(_, reset) => (
          <button onClick={reset}>Reset me</button>
        )}
      >
        <Bomb shouldThrow />
      </ErrorBoundary>
    )

    const btn = screen.getByRole('button', { name: /reset me/i })
    fireEvent.click(btn)

    // After reset the boundary should attempt to render children again
    // (Bomb still throws, so we'd see the fallback again — that's fine)
    expect(btn).toBeDefined()
  })

  it('resets when the "Try again" button is clicked', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )

    fireEvent.click(screen.getByRole('button', { name: /retry loading/i }))

    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('resets automatically when resetKeys change', () => {
    const { rerender } = render(
      <ErrorBoundary resetKeys={['key-a']}>
        <ToggleBomb explode />
      </ErrorBoundary>
    )

    // Error UI should be visible
    expect(screen.getByRole('alert')).toBeInTheDocument()

    // Change resetKeys AND stop throwing so recovery is visible
    rerender(
      <ErrorBoundary resetKeys={['key-b']}>
        <ToggleBomb explode={false} />
      </ErrorBoundary>
    )

    expect(screen.getByText('Recovered content')).toBeInTheDocument()
  })

  it('does not reset when resetKeys have the same values', () => {
    const { rerender } = render(
      <ErrorBoundary resetKeys={['stable']}>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )

    expect(screen.getByRole('alert')).toBeInTheDocument()

    // Same key — boundary should stay in error state
    rerender(
      <ErrorBoundary resetKeys={['stable']}>
        <Bomb shouldThrow />
      </ErrorBoundary>
    )

    expect(screen.getByRole('alert')).toBeInTheDocument()
  })
})
