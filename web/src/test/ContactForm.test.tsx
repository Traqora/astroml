import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { ContactForm } from '../components/ContactForm'
import { post, ApiError } from '../api/client'

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, post: vi.fn() }
})

const mockedPost = post as unknown as ReturnType<typeof vi.fn>

function fill(values: Partial<Record<string, string>>) {
  if (values.name !== undefined)
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: values.name } })
  if (values.email !== undefined)
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: values.email } })
  if (values.subject !== undefined)
    fireEvent.change(screen.getByLabelText('Subject'), { target: { value: values.subject } })
  if (values.message !== undefined)
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: values.message } })
}

describe('ContactForm', () => {
  beforeEach(() => {
    mockedPost.mockReset()
  })

  test('renders all fields', () => {
    render(<ContactForm />)
    expect(screen.getByLabelText('Name')).toBeInTheDocument()
    expect(screen.getByLabelText('Email')).toBeInTheDocument()
    expect(screen.getByLabelText('Subject')).toBeInTheDocument()
    expect(screen.getByLabelText('Message')).toBeInTheDocument()
  })

  test('shows validation errors and does not submit when empty', () => {
    render(<ContactForm />)
    fireEvent.click(screen.getByRole('button', { name: /send message/i }))
    expect(screen.getByText('Name is required')).toBeInTheDocument()
    expect(screen.getByText('Email is required')).toBeInTheDocument()
    expect(screen.getByText('Subject is required')).toBeInTheDocument()
    expect(screen.getByText('Message is required')).toBeInTheDocument()
    expect(mockedPost).not.toHaveBeenCalled()
  })

  test('rejects an invalid email address', () => {
    render(<ContactForm />)
    fill({ name: 'Ada', email: 'bad-email', subject: 'Hi', message: 'Hello there' })
    fireEvent.click(screen.getByRole('button', { name: /send message/i }))
    expect(screen.getByText('Enter a valid email address')).toBeInTheDocument()
    expect(mockedPost).not.toHaveBeenCalled()
  })

  test('submits valid data and shows the ticket reference', async () => {
    mockedPost.mockResolvedValue({
      message: 'received',
      ticket: { reference: 'TKT-ABCD1234', status: 'open', created_at: '2026-01-01T00:00:00Z' },
    })

    render(<ContactForm />)
    fill({ name: 'Ada', email: 'ada@example.com', subject: 'Help', message: 'I need help.' })
    fireEvent.click(screen.getByRole('button', { name: /send message/i }))

    await waitFor(() =>
      expect(screen.getByTestId('ticket-reference')).toHaveTextContent('TKT-ABCD1234'),
    )
    expect(mockedPost).toHaveBeenCalledWith(
      '/api/v1/contact',
      expect.objectContaining({
        name: 'Ada',
        email: 'ada@example.com',
        subject: 'Help',
        message: 'I need help.',
      }),
    )
  })

  test('surfaces a server error', async () => {
    mockedPost.mockRejectedValue(new ApiError(503, 'Service unavailable'))

    render(<ContactForm />)
    fill({ name: 'Ada', email: 'ada@example.com', subject: 'Help', message: 'I need help.' })
    fireEvent.click(screen.getByRole('button', { name: /send message/i }))

    await waitFor(() => expect(screen.getByText('Service unavailable')).toBeInTheDocument())
  })
})
