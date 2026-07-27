import { render, screen, fireEvent, act } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { AdvancedSearch } from '../components/AdvancedSearch'
import type { AdvancedSearchFilters } from '../components/AdvancedSearch'

const storageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => { store[k] = v },
    removeItem: (k: string) => { delete store[k] },
    clear: () => { store = {} },
  }
})()

Object.defineProperty(window, 'localStorage', { value: storageMock })

describe('AdvancedSearch', () => {
  beforeEach(() => {
    storageMock.clear()
  })

  it('renders the search input with placeholder', () => {
    render(<AdvancedSearch onSearch={() => {}} placeholder="Search here…" />)
    expect(screen.getByPlaceholderText('Search here…')).toBeInTheDocument()
  })

  it('calls onSearch with debounced query when input changes', async () => {
    vi.useFakeTimers()
    const onSearch = vi.fn()
    render(<AdvancedSearch onSearch={onSearch} debounceMs={300} />)

    fireEvent.change(screen.getByRole('textbox', { name: /search query/i }), {
      target: { value: 'GABCDE' },
    })

    expect(onSearch).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(onSearch).toHaveBeenCalledWith(expect.objectContaining({ query: 'GABCDE' }))
    vi.useRealTimers()
  })

  it('calls onSearch with date range filters', async () => {
    vi.useFakeTimers()
    const onSearch = vi.fn()
    render(<AdvancedSearch onSearch={onSearch} debounceMs={0} />)

    fireEvent.change(screen.getByLabelText(/start date/i), {
      target: { value: '2024-01-01' },
    })
    fireEvent.change(screen.getByLabelText(/end date/i), {
      target: { value: '2024-06-30' },
    })

    await act(async () => { vi.advanceTimersByTime(0) })

    const calls = onSearch.mock.calls
    const lastCall: AdvancedSearchFilters = calls[calls.length - 1][0]
    expect(lastCall.startDate).toBe('2024-01-01')
    expect(lastCall.endDate).toBe('2024-06-30')
    vi.useRealTimers()
  })

  it('saves and loads search history to localStorage on Enter', async () => {
    vi.useFakeTimers()
    const onSearch = vi.fn()
    render(<AdvancedSearch onSearch={onSearch} debounceMs={0} />)

    const input = screen.getByRole('textbox', { name: /search query/i })
    fireEvent.change(input, { target: { value: 'test-account' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    const stored = JSON.parse(storageMock.getItem('astroml:search:history') ?? '[]') as string[]
    expect(stored).toContain('test-account')
    vi.useRealTimers()
  })

  it('shows autocomplete suggestions matching the query', () => {
    const suggestions = [
      { label: 'GABCDE', value: 'GABCDE', type: 'account' as const },
      { label: 'GXYZ12', value: 'GXYZ12', type: 'account' as const },
    ]
    render(<AdvancedSearch onSearch={() => {}} suggestions={suggestions} />)

    const input = screen.getByRole('textbox', { name: /search query/i })
    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'GAB' } })

    expect(screen.getByText('GABCDE')).toBeInTheDocument()
    expect(screen.queryByText('GXYZ12')).not.toBeInTheDocument()
  })
})
