import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { AccountSearchBar } from '../components/AccountSearchBar/AccountSearchBar'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../api/search', async () => {
  const actual = await vi.importActual<typeof import('../api/search')>('../api/search')
  return {
    ...actual,
    searchAccounts: vi.fn(),
    getAutocompleteSuggestions: vi.fn(),
  }
})

import { searchAccounts, getAutocompleteSuggestions } from '../api/search'

const mockedSearchAccounts = searchAccounts as unknown as ReturnType<typeof vi.fn>
const mockedGetAutocompleteSuggestions = getAutocompleteSuggestions as unknown as ReturnType<typeof vi.fn>

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  )
}

describe('AccountSearchBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders search input with placeholder', () => {
    renderWithProviders(<AccountSearchBar />)
    expect(screen.getByRole('combobox')).toHaveAttribute('placeholder', 'Search accounts, transactions...')
  })

  test('debounces search input and calls API', async () => {
    mockedSearchAccounts.mockResolvedValue({
      query: 'test',
      mode: 'hybrid',
      results: [
        {
          id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
          title: 'Test Account',
          content: 'Account details',
          type: 'account',
          score: 0.95,
          method: 'semantic',
          metadata: {},
        },
      ],
    })

    renderWithProviders(<AccountSearchBar />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'test' } })

    await waitFor(() => {
      expect(mockedSearchAccounts).toHaveBeenCalledWith('test', { limit: 10 })
    }, { timeout: 500 })
  })

  test('shows autocomplete suggestions for short queries', async () => {
    mockedGetAutocompleteSuggestions.mockResolvedValue({
      suggestions: ['test account'],
    })

    renderWithProviders(<AccountSearchBar />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'te' } })

    await waitFor(() => {
      expect(mockedGetAutocompleteSuggestions).toHaveBeenCalledWith('te')
    }, { timeout: 500 })
  })

  test('shows no results message when search returns empty', async () => {
    mockedSearchAccounts.mockResolvedValue({
      query: 'nonexistent',
      mode: 'hybrid',
      results: [],
    })

    renderWithProviders(<AccountSearchBar />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'nonexistent' } })

    await waitFor(() => {
      expect(screen.getByText('No results found')).toBeInTheDocument()
    }, { timeout: 500 })
  })

  test('calls onSelectResult when a result is clicked', async () => {
    const onSelectResult = vi.fn()
    mockedSearchAccounts.mockResolvedValue({
      query: 'test',
      mode: 'hybrid',
      results: [
        {
          id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
          title: 'Test Account',
          content: 'Account details',
          type: 'account',
          score: 0.95,
          method: 'semantic',
          metadata: {},
        },
      ],
    })

    renderWithProviders(<AccountSearchBar onSelectResult={onSelectResult} />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'test' } })

    await waitFor(() => {
      expect(screen.getByText('Test Account')).toBeInTheDocument()
    }, { timeout: 500 })

    fireEvent.mouseDown(screen.getByText('Test Account'))

    expect(onSelectResult).toHaveBeenCalledWith({
      id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
      type: 'account',
      title: 'Test Account',
    })
  })

  test('supports keyboard navigation', async () => {
    mockedSearchAccounts.mockResolvedValue({
      query: 'test',
      mode: 'hybrid',
      results: [
        {
          id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
          title: 'Test Account',
          content: 'Account details',
          type: 'account',
          score: 0.95,
          method: 'semantic',
          metadata: {},
        },
      ],
    })

    renderWithProviders(<AccountSearchBar />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'test' } })

    await waitFor(() => {
      expect(screen.getByText('Test Account')).toBeInTheDocument()
    }, { timeout: 500 })

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    fireEvent.keyDown(input, { key: 'Escape' })
  })

  test('hides results on Escape key', async () => {
    mockedSearchAccounts.mockResolvedValue({
      query: 'test',
      mode: 'hybrid',
      results: [
        {
          id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
          title: 'Test Account',
          content: 'Account details',
          type: 'account',
          score: 0.95,
          method: 'semantic',
          metadata: {},
        },
      ],
    })

    renderWithProviders(<AccountSearchBar />)

    const input = screen.getByRole('combobox')
    fireEvent.change(input, { target: { value: 'test' } })

    await waitFor(() => {
      expect(screen.getByText('Test Account')).toBeInTheDocument()
    }, { timeout: 500 })

    fireEvent.keyDown(input, { key: 'Escape' })

    await waitFor(() => {
      expect(screen.queryByText('Test Account')).not.toBeInTheDocument()
    })
  })
})
