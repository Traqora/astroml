import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import { OnboardingChecklist } from '../components/OnboardingChecklist'

const mockProgress = {
  github_username: 'octocat',
  checklist: [
    { step: 'fork_repo', label: 'Fork the repository', completed: true },
    { step: 'setup_dev_environment', label: 'Set up dev environment', completed: false },
    { step: 'run_tests', label: 'Run the test suite', completed: false },
    { step: 'first_pr', label: 'Open your first pull request', completed: false },
  ],
  completed_count: 1,
  total_steps: 4,
  progress_pct: 25,
  is_complete: false,
  started_at: '2026-01-01T00:00:00Z',
  last_updated: '2026-01-01T00:00:00Z',
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => mockProgress,
  }))
})

afterEach(() => {
  vi.restoreAllMocks()
})

test('renders checklist items', async () => {
  render(<OnboardingChecklist githubUsername="octocat" />)
  await waitFor(() => expect(screen.getByText('Fork the repository')).toBeInTheDocument())
  expect(screen.getByText('Set up dev environment')).toBeInTheDocument()
  expect(screen.getByText('Run the test suite')).toBeInTheDocument()
  expect(screen.getByText('Open your first pull request')).toBeInTheDocument()
})

test('shows progress percentage', async () => {
  render(<OnboardingChecklist githubUsername="octocat" />)
  await waitFor(() => expect(screen.getByText(/25%/)).toBeInTheDocument())
})

test('completed step has no mark-done button', async () => {
  render(<OnboardingChecklist githubUsername="octocat" />)
  await waitFor(() => screen.getByText('Fork the repository'))
  const buttons = screen.getAllByRole('button', { name: /mark done/i })
  expect(buttons).toHaveLength(3)
})

test('clicking mark done calls complete endpoint', async () => {
  const afterComplete = { ...mockProgress, completed_count: 2, progress_pct: 50 }
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => mockProgress })
    .mockResolvedValueOnce({ ok: true, json: async () => afterComplete })
  vi.stubGlobal('fetch', fetchMock)

  render(<OnboardingChecklist githubUsername="octocat" />)
  await waitFor(() => screen.getAllByRole('button', { name: /mark done/i }))
  fireEvent.click(screen.getAllByRole('button', { name: /mark done/i })[0])
  await waitFor(() => expect(screen.getByText(/50%/)).toBeInTheDocument())
})

test('shows complete badge when all steps done', async () => {
  const completeProgress = {
    ...mockProgress,
    checklist: mockProgress.checklist.map((i) => ({ ...i, completed: true })),
    completed_count: 4,
    progress_pct: 100,
    is_complete: true,
  }
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => completeProgress }))
  render(<OnboardingChecklist githubUsername="octocat" />)
  await waitFor(() => expect(screen.getByText(/onboarding complete/i)).toBeInTheDocument())
})
