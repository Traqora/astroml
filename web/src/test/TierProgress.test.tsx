import { render, screen } from '@testing-library/react'
import * as confetti from '../lib/confetti'
import { TierProgress } from '../components/LoyaltyDashboard/TierProgress'

vi.spyOn(confetti, 'fireConfetti').mockImplementation(() => {})

const gold = { id: 'gold', name: 'Gold', threshold: 3000, multiplier: 1.25, color: '#d4af37' }
const platinum = { id: 'platinum', name: 'Platinum', threshold: 6000, multiplier: 1.5, color: '#e5e4e2' }

test('renders progress and remaining', () => {
  render(<TierProgress currentTier={gold} nextTier={{ tier: platinum, remainingToUpgrade: 1000, progressPct: 75 }} />)
  expect(screen.getByText('75%')).toBeInTheDocument()
  expect(screen.getByText(/1000 points to Platinum/i)).toBeInTheDocument()
})

test('fires confetti when tier changes', async () => {
  const { rerender } = render(<TierProgress currentTier={gold} nextTier={{ tier: platinum, remainingToUpgrade: 1000, progressPct: 75 }} />)
  rerender(<TierProgress currentTier={platinum} nextTier={undefined} />)
  expect(confetti.fireConfetti).toHaveBeenCalled()
})
