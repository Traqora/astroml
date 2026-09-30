import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ModelMonitoringDashboard } from '../components/ModelMonitoringDashboard'
import * as clientApi from '../api/client'

vi.spyOn(clientApi, 'get').mockResolvedValue({
  accuracy: 0.93,
  f1: 0.86,
  drift_score: 0.12,
  auc: 0.91,
  performance: [
    { date: '2026-04-01', accuracy: 0.88, drift: 0.08 },
    { date: '2026-04-08', accuracy: 0.91, drift: 0.10 },
  ],
})

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

test('renders model monitoring dashboard', async () => {
  renderWithClient(<ModelMonitoringDashboard />)

  await waitFor(() => expect(screen.getByText(/Model Performance/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Prediction Accuracy Trend/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Data Drift/i)).toBeInTheDocument())
})
