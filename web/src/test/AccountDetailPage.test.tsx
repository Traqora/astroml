import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { AccountDetailPage } from '../components/AccountDetailPage/AccountDetailPage'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const mockAccount = {
  account_id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
  balance: 1000.5,
  sequence: 12345,
  home_domain: 'example.com',
  flags: 0,
  last_modified_ledger: 67890,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
}

const mockFraud = {
  account_id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
  total_alerts: 3,
  high_risk: 1,
  medium_risk: 1,
  low_risk: 1,
  latest_score: 0.75,
}

const mockLoyalty = {
  account_id: 'GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
  points_balance: 5000,
  tier_id: 'gold',
  tier_name: 'Gold',
}

vi.mock('../api/accounts', async () => {
  const actual = await vi.importActual<typeof import('../api/accounts')>('../api/accounts')
  return {
    ...actual,
    getAccountDetail: vi.fn(),
    getAccountFraudSummary: vi.fn(),
    getAccountLoyaltySummary: vi.fn(),
    getAccountTransactions: vi.fn(),
  }
})

import {
  getAccountDetail,
  getAccountFraudSummary,
  getAccountLoyaltySummary,
  getAccountTransactions,
} from '../api/accounts'

const mockedGetAccountDetail = getAccountDetail as unknown as ReturnType<typeof vi.fn>
const mockedGetAccountFraudSummary = getAccountFraudSummary as unknown as ReturnType<typeof vi.fn>
const mockedGetAccountLoyaltySummary = getAccountLoyaltySummary as unknown as ReturnType<typeof vi.fn>
const mockedGetAccountTransactions = getAccountTransactions as unknown as ReturnType<typeof vi.fn>

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  )
}

describe('AccountDetailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders account details when data loads', async () => {
    mockedGetAccountDetail.mockResolvedValue(mockAccount)
    mockedGetAccountFraudSummary.mockResolvedValue(mockFraud)
    mockedGetAccountLoyaltySummary.mockResolvedValue(mockLoyalty)
    mockedGetAccountTransactions.mockResolvedValue({
      data: [],
      page: 1,
      pageSize: 10,
      total: 0,
    })

    renderWithProviders(<AccountDetailPage publicKey={mockAccount.account_id} />)

    await waitFor(() => {
      expect(screen.getByText('Account Details')).toBeInTheDocument()
    })

    expect(screen.getByText('1,000.5 XLM')).toBeInTheDocument()
    expect(screen.getByText('12345')).toBeInTheDocument()
    expect(screen.getByText('example.com')).toBeInTheDocument()
  })

  test('renders not found when account is null', async () => {
    mockedGetAccountDetail.mockResolvedValue(null)
    mockedGetAccountFraudSummary.mockResolvedValue(null)
    mockedGetAccountLoyaltySummary.mockResolvedValue(null)
    mockedGetAccountTransactions.mockResolvedValue({
      data: [],
      page: 1,
      pageSize: 10,
      total: 0,
    })

    renderWithProviders(<AccountDetailPage publicKey="GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890" />)

    await waitFor(() => {
      expect(screen.getByText('Account not found')).toBeInTheDocument()
    })
  })

  test('renders fraud and loyalty summaries', async () => {
    mockedGetAccountDetail.mockResolvedValue(mockAccount)
    mockedGetAccountFraudSummary.mockResolvedValue(mockFraud)
    mockedGetAccountLoyaltySummary.mockResolvedValue(mockLoyalty)
    mockedGetAccountTransactions.mockResolvedValue({
      data: [],
      page: 1,
      pageSize: 10,
      total: 0,
    })

    renderWithProviders(<AccountDetailPage publicKey={mockAccount.account_id} />)

    await waitFor(() => {
      expect(screen.getByText('Fraud Summary')).toBeInTheDocument()
    })

    expect(screen.getByText('Gold')).toBeInTheDocument()
    expect(screen.getByText('5,000')).toBeInTheDocument()
  })
})
