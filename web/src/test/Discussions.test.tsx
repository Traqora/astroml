/**
 * ARIA/keyboard accessibility tests for Discussions — issue #850.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { Discussions } from '../components/Discussions/Discussions'
import * as apiClient from '../api/client'

const DISCUSSION = {
  id: 'd1',
  title: 'Why is the mempool so slow',
  body: 'Investigating latency.',
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:00Z',
  author: 'octocat',
  category: 'general',
  commentCount: 2,
}

const REPUTATION = {
  username: 'octocat',
  reputationScore: 42,
  contributions: { commits: 10, issues: 2, pullRequests: 3 },
  followers: 5,
  repositories: 7,
}

function mockApi() {
  vi.spyOn(apiClient, 'get').mockImplementation(async (endpoint: string) => {
    if (endpoint.startsWith('/api/v1/discussions/recent')) {
      return { discussions: [DISCUSSION] } as never
    }
    if (endpoint.startsWith('/api/v1/discussions/categories')) {
      return { categories: [] } as never
    }
    if (endpoint.startsWith('/api/v1/discussions/user-reputation/')) {
      return REPUTATION as never
    }
    throw new Error(`Unexpected endpoint: ${endpoint}`)
  })
}

async function renderDiscussionsAndWait() {
  render(<Discussions />)
  await waitFor(() => screen.getByText(DISCUSSION.title))
}

describe('Discussions accessibility', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('exposes the author control as a keyboard-reachable button', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    const authorControl = screen.getByText(/By/).closest('span')
    expect(authorControl).toHaveAttribute('role', 'button')
    expect(authorControl).toHaveAttribute('tabIndex', '0')
  })

  it('opens the reputation modal on Enter key, not just click', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    const authorControl = screen.getByText(/By/).closest('span')!
    fireEvent.keyDown(authorControl, { key: 'Enter' })

    await waitFor(() => expect(screen.getByRole('heading', { name: 'octocat' })).toBeInTheDocument())
  })

  it('opens the reputation modal on Space key', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    const authorControl = screen.getByText(/By/).closest('span')!
    fireEvent.keyDown(authorControl, { key: ' ' })

    await waitFor(() => expect(screen.getByRole('heading', { name: 'octocat' })).toBeInTheDocument())
  })

  it('marks the reputation modal with dialog semantics', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    fireEvent.click(screen.getByText(/By/))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'octocat' })).toBeInTheDocument())

    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveAttribute('aria-labelledby', 'reputation-modal-title')
  })

  it('focuses the close button when the modal opens', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    fireEvent.click(screen.getByText(/By/))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'octocat' })).toBeInTheDocument())

    expect(screen.getByRole('button', { name: 'Close' })).toHaveFocus()
  })

  it('closes the reputation modal on Escape', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    fireEvent.click(screen.getByText(/By/))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'octocat' })).toBeInTheDocument())

    fireEvent.keyDown(document, { key: 'Escape' })

    await waitFor(() => expect(screen.queryByRole('heading', { name: 'octocat' })).not.toBeInTheDocument())
  })

  it('does not close on Escape when no modal is open', async () => {
    mockApi()
    await renderDiscussionsAndWait()

    expect(() => fireEvent.keyDown(document, { key: 'Escape' })).not.toThrow()
  })
})
