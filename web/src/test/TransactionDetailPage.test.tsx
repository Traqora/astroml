import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { TransactionDetailPage } from '../components/TransactionDetailPage/TransactionDetailPage'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const mockTransaction = {
  hash: 'abc123def456789012345',
  ledgerSequence: 12345,
  sourceAccount: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
  destinationAccount: 'GDEF1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
  amount: 100.5,
  assetCode: 'XLM',
  assetIssuer: null,
  operationType: 'payment',
  createdAt: '2024-01-01T00:00:00Z',
  fee: 100,
  successful: true,
  memoType: 'text',
}

vi.mock('../api/transactions', async () => {
  const actual = await vi.importActual<typeof import('../api/transactions')>('../api/transactions')
  return {
    ...actual,
    getTransactionByHash: vi.fn(),
  }
})

import { getTransactionByHash } from '../api/transactions'

const mockedGetTransactionByHash = getTransactionByHash as unknown as ReturnType<typeof vi.fn>

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  )
}

describe('TransactionDetailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders transaction details when data loads', async () => {
    mockedGetTransactionByHash.mockResolvedValue(mockTransaction)

    renderWithProviders(<TransactionDetailPage hash={mockTransaction.hash} />)

    await waitFor(() => {
      expect(screen.getByText('Transaction Details')).toBeInTheDocument()
    })

    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('payment')).toBeInTheDocument()
    expect(screen.getByText('100.5 XLM')).toBeInTheDocument()
    expect(screen.getByText('100 stroops')).toBeInTheDocument()
  })

  test('renders not found when transaction is null', async () => {
    mockedGetTransactionByHash.mockResolvedValue(null)

    renderWithProviders(<TransactionDetailPage hash="nonexistent" />)

    await waitFor(() => {
      expect(screen.getByText('Transaction not found')).toBeInTheDocument()
    })
  })

  test('renders source and destination accounts', async () => {
    mockedGetTransactionByHash.mockResolvedValue(mockTransaction)

    renderWithProviders(<TransactionDetailPage hash={mockTransaction.hash} />)

    await waitFor(() => {
      expect(screen.getByText(mockTransaction.sourceAccount)).toBeInTheDocument()
    })

    expect(screen.getByText(mockTransaction.destinationAccount!)).toBeInTheDocument()
  })

  test('renders failed status correctly', async () => {
    mockedGetTransactionByHash.mockResolvedValue({ ...mockTransaction, successful: false })

    renderWithProviders(<TransactionDetailPage hash={mockTransaction.hash} />)

    await waitFor(() => {
      expect(screen.getByText('Failed')).toBeInTheDocument()
    })
  })
})
