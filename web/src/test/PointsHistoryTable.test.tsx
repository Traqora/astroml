import { render, screen } from '@testing-library/react'
import { PointsHistoryTable } from '../components/LoyaltyDashboard/PointsHistoryTable'

const response = {
  data: [
    { id: '1', date: new Date().toISOString(), type: 'earn', points: 100, source: 'Purchase' },
    { id: '2', date: new Date().toISOString(), type: 'redeem', points: -50, source: 'Redemption' },
  ],
  page: 0,
  pageSize: 10,
  total: 2,
}

test('renders history rows', () => {
  render(
    <PointsHistoryTable response={response as any} loading={false} page={0} pageSize={10} onPageChange={() => {}} />
  )
  expect(screen.getByText('Purchase')).toBeInTheDocument()
  expect(screen.getByText('Redemption')).toBeInTheDocument()
})
