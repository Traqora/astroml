/**
 * Keyboard/focus-management tests for the dashboard's NavBar — issue #788.
 */
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import type { ReactNode } from 'react'
import { NavBar } from '../App'
import { ThemeProvider } from '../contexts/ThemeContext'

function renderNavBar(children?: ReactNode) {
  return render(
    <ThemeProvider>
      <NavBar view={{ kind: 'home' }} onNavigate={vi.fn()} />
      {children}
    </ThemeProvider>
  )
}

function mockMatchMedia(mobileMatches: boolean) {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (query: string) =>
      ({
        matches: query.includes('max-width: 640px') ? mobileMatches : false,
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList
  )
}

describe('NavBar', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('labels the primary navigation landmark', () => {
    mockMatchMedia(false)
    renderNavBar()

    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
  })

  it('exposes menu state via aria-expanded/aria-haspopup/aria-controls on the mobile toggle', () => {
    mockMatchMedia(true)
    renderNavBar()

    const toggle = screen.getByRole('button', { name: /toggle navigation menu/i })
    expect(toggle).toHaveAttribute('aria-haspopup', 'true')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')

    fireEvent.click(toggle)

    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    const controlsId = toggle.getAttribute('aria-controls')
    expect(controlsId).toBeTruthy()
    expect(document.getElementById(controlsId as string)).toBeInTheDocument()
  })

  it('closes the mobile menu and returns focus to the toggle on Escape', () => {
    mockMatchMedia(true)
    renderNavBar()

    const toggle = screen.getByRole('button', { name: /toggle navigation menu/i })
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')

    fireEvent.keyDown(document, { key: 'Escape' })

    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(document.activeElement).toBe(toggle)
  })

  it('closes the mobile menu on an outside click', () => {
    mockMatchMedia(true)
    renderNavBar(<div data-testid="outside">outside</div>)

    const toggle = screen.getByRole('button', { name: /toggle navigation menu/i })
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')

    fireEvent.mouseDown(screen.getByTestId('outside'))

    expect(toggle).toHaveAttribute('aria-expanded', 'false')
  })
})
