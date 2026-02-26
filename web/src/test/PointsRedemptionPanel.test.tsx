import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { PointsRedemptionPanel } from '../components/LoyaltyDashboard/PointsRedemptionPanel'

function setup(balance = 1000) {
  const client = new QueryClient()
  const onRedeem = vi.fn()
  render(
    <QueryClientProvider client={client}>
      <PointsRedemptionPanel balance={balance} onRedeem={onRedeem} pending={false} />
    </QueryClientProvider>
  )
  return { onRedeem }
}

test('validates redemption amount and calls onRedeem', async () => {
  const { onRedeem } = setup(500)
  const input = screen.getByRole('spinbutton')
  fireEvent.change(input, { target: { value: '200' } })
  fireEvent.click(screen.getByText('Redeem'))
  expect(onRedeem).toHaveBeenCalledWith(200)
})

test('shows error when amount exceeds balance', async () => {
  setup(100)
  const input = screen.getByRole('spinbutton')
  fireEvent.change(input, { target: { value: '200' } })
  fireEvent.click(screen.getByText('Redeem'))
  expect(await screen.findByText(/exceeds balance/i)).toBeInTheDocument()
})
