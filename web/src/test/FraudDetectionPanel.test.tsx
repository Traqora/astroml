import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { FraudDetectionPanel } from '../components/LoyaltyDashboard/FraudDetectionPanel'
import * as loyaltyApi from '../api/loyalty'

vi.spyOn(loyaltyApi, 'getFraudStats').mockResolvedValue({
  totalAlerts: 10,
  highRisk: 3,
  mediumRisk: 4,
  lowRisk: 3,
  recentAlerts: [
    { id: '1', accountId: 'GA123', pattern: 'sybil_cluster' as const, riskScore: 85, detectedAt: '2026-01-01', description: 'Test alert' },
  ],
  riskOverTime: [
    { date: '2026-01-01', score: 50 },
    { date: '2026-01-02', score: 60 },
  ],
})

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

test('renders fraud stats and recent alerts', async () => {
  renderWithClient(<FraudDetectionPanel />)
  await waitFor(() => expect(screen.getByText(/Fraud Detection/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Total Alerts/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/High Risk/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Recent Alerts/i)).toBeInTheDocument())
})

test('renders risk distribution labels', async () => {
  renderWithClient(<FraudDetectionPanel />)
  await waitFor(() => expect(screen.getByText(/Risk Distribution/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Risk Score/i)).toBeInTheDocument())
})
