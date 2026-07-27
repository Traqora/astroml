/**
 * Feedback (issue #308)
 *
 * In-app feedback panel for bug reports, feature requests, and general
 * comments. Supports an optional screenshot attachment (read as a data URL)
 * and submits to the feedback API.
 */
import React, { useState } from 'react'
import { post, ApiError } from '../api/client'

type Category = 'bug' | 'feature' | 'general'

const CATEGORIES: { value: Category; label: string }[] = [
  { value: 'bug', label: 'Bug report' },
  { value: 'feature', label: 'Feature request' },
  { value: 'general', label: 'General feedback' },
]

interface FeedbackResponse {
  id: number
  category: string
  status: string
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export function Feedback() {
  const [category, setCategory] = useState<Category>('bug')
  const [message, setMessage] = useState('')
  const [email, setEmail] = useState('')
  const [screenshot, setScreenshot] = useState<string | undefined>()
  const [error, setError] = useState('')
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success'>('idle')

  async function handleScreenshot(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) {
      setScreenshot(undefined)
      return
    }
    try {
      setScreenshot(await readFileAsDataUrl(file))
    } catch {
      setScreenshot(undefined)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    if (!message.trim()) {
      setError('Please enter your feedback')
      return
    }

    setStatus('submitting')
    try {
      await post<FeedbackResponse>('/api/v1/feedback', {
        category,
        message: message.trim(),
        email: email.trim() || undefined,
        screenshot,
      })
      setStatus('success')
      setMessage('')
      setEmail('')
      setScreenshot(undefined)
    } catch (err) {
      setStatus('idle')
      setError(
        err instanceof ApiError ? err.message : 'Could not submit feedback. Please try again.',
      )
    }
  }

  if (status === 'success') {
    return (
      <div className="feedback feedback--success" role="status">
        <h3>Thanks for your feedback!</h3>
        <p>We review every submission to help shape the roadmap.</p>
        <button type="button" onClick={() => setStatus('idle')}>
          Send more feedback
        </button>
      </div>
    )
  }

  const submitting = status === 'submitting'

  return (
    <form className="feedback" onSubmit={handleSubmit} aria-label="Feedback form">
      <h2>Send feedback</h2>

      <fieldset className="feedback__categories">
        <legend>Category</legend>
        {CATEGORIES.map((c) => (
          <label key={c.value}>
            <input
              type="radio"
              name="feedback-category"
              value={c.value}
              checked={category === c.value}
              onChange={() => setCategory(c.value)}
              disabled={submitting}
            />
            {c.label}
          </label>
        ))}
      </fieldset>

      <div className="feedback__field">
        <label htmlFor="feedback-message">Your feedback</label>
        <textarea
          id="feedback-message"
          rows={5}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          aria-invalid={error ? 'true' : undefined}
          disabled={submitting}
        />
      </div>

      <div className="feedback__field">
        <label htmlFor="feedback-email">Email (optional)</label>
        <input
          id="feedback-email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={submitting}
        />
      </div>

      <div className="feedback__field">
        <label htmlFor="feedback-screenshot">Screenshot (optional)</label>
        <input
          id="feedback-screenshot"
          type="file"
          accept="image/*"
          onChange={handleScreenshot}
          disabled={submitting}
        />
        {screenshot && <span className="feedback__attached">Screenshot attached</span>}
      </div>

      {error && <p className="feedback__error" role="alert">{error}</p>}

      <button type="submit" disabled={submitting}>
        {submitting ? 'Sending…' : 'Send feedback'}
      </button>
    </form>
  )
}

export default Feedback
