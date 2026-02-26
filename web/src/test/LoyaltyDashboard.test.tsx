import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from '../App'

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient()
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

test('renders dashboard sections', async () => {
  renderWithClient(<App />)

  await waitFor(() => expect(screen.getByText(/Loyalty Dashboard/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Current Tier/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Points Balance/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Tier Benefits/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Redeem Points/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Tier Comparison/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Points History/i)).toBeInTheDocument())
  await waitFor(() => expect(screen.getByText(/Invite Friends/i)).toBeInTheDocument())
})
