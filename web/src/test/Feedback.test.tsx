import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { Feedback } from '../components/Feedback'
import { post, ApiError } from '../api/client'

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, post: vi.fn() }
})

const mockedPost = post as unknown as ReturnType<typeof vi.fn>

describe('Feedback', () => {
  beforeEach(() => {
    mockedPost.mockReset()
  })

  test('renders category options and the message field', () => {
    render(<Feedback />)
    expect(screen.getByLabelText('Bug report')).toBeInTheDocument()
    expect(screen.getByLabelText('Feature request')).toBeInTheDocument()
    expect(screen.getByLabelText('General feedback')).toBeInTheDocument()
    expect(screen.getByLabelText('Your feedback')).toBeInTheDocument()
  })

  test('blocks submit when the message is empty', () => {
    render(<Feedback />)
    fireEvent.click(screen.getByRole('button', { name: /send feedback/i }))
    expect(screen.getByText('Please enter your feedback')).toBeInTheDocument()
    expect(mockedPost).not.toHaveBeenCalled()
  })

  test('submits the selected category and message, then shows success', async () => {
    mockedPost.mockResolvedValue({ id: 1, category: 'feature', status: 'open' })

    render(<Feedback />)
    fireEvent.click(screen.getByLabelText('Feature request'))
    fireEvent.change(screen.getByLabelText('Your feedback'), {
      target: { value: 'Please add dark mode' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send feedback/i }))

    await waitFor(() => expect(screen.getByText(/thanks for your feedback/i)).toBeInTheDocument())
    expect(mockedPost).toHaveBeenCalledWith(
      '/api/v1/feedback',
      expect.objectContaining({ category: 'feature', message: 'Please add dark mode' }),
    )
  })

  test('surfaces a server error', async () => {
    mockedPost.mockRejectedValue(new ApiError(500, 'Server exploded'))

    render(<Feedback />)
    fireEvent.change(screen.getByLabelText('Your feedback'), {
      target: { value: 'Something broke' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send feedback/i }))

    await waitFor(() => expect(screen.getByText('Server exploded')).toBeInTheDocument())
  })
})
