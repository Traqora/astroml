/**
 * ContactForm (issue #305)
 *
 * Lets users reach out for support. Validates input client-side, submits to the
 * contact API, and on success shows the generated support-ticket reference.
 *
 * reCAPTCHA is optional: when `VITE_RECAPTCHA_SITE_KEY` is configured and the
 * grecaptcha script is loaded, a token is attached to the submission. The
 * backend skips verification when no secret is configured, so the form works in
 * development without a key.
 */
import React, { useState } from 'react'
import { post, ApiError } from '../api/client'

interface ContactFormState {
  name: string
  email: string
  subject: string
  message: string
}

interface SupportTicket {
  reference: string
  status: string
  created_at: string
}

interface ContactSubmitResponse {
  message: string
  ticket: SupportTicket
}

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

const EMPTY: ContactFormState = { name: '', email: '', subject: '', message: '' }

function validate(form: ContactFormState): Partial<Record<keyof ContactFormState, string>> {
  const errors: Partial<Record<keyof ContactFormState, string>> = {}
  if (!form.name.trim()) errors.name = 'Name is required'
  if (!form.email.trim()) errors.email = 'Email is required'
  else if (!EMAIL_RE.test(form.email.trim())) errors.email = 'Enter a valid email address'
  if (!form.subject.trim()) errors.subject = 'Subject is required'
  if (!form.message.trim()) errors.message = 'Message is required'
  return errors
}

/** Reads a reCAPTCHA token if the widget is configured and loaded. */
function getRecaptchaToken(): string | undefined {
  const grecaptcha = (window as unknown as { grecaptcha?: { getResponse?: () => string } })
    .grecaptcha
  const token = grecaptcha?.getResponse?.()
  return token || undefined
}

export function ContactForm() {
  const [form, setForm] = useState<ContactFormState>(EMPTY)
  const [errors, setErrors] = useState<Partial<Record<keyof ContactFormState, string>>>({})
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle')
  const [serverError, setServerError] = useState('')
  const [ticketRef, setTicketRef] = useState('')

  const recaptchaSiteKey = import.meta.env.VITE_RECAPTCHA_SITE_KEY as string | undefined

  function update(field: keyof ContactFormState, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }))
    setErrors((prev) => ({ ...prev, [field]: undefined }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setServerError('')
    const validationErrors = validate(form)
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors)
      return
    }

    setStatus('submitting')
    try {
      const res = await post<ContactSubmitResponse>('/api/v1/contact', {
        name: form.name.trim(),
        email: form.email.trim(),
        subject: form.subject.trim(),
        message: form.message.trim(),
        recaptcha_token: getRecaptchaToken(),
      })
      setTicketRef(res.ticket.reference)
      setStatus('success')
      setForm(EMPTY)
    } catch (err) {
      setStatus('error')
      setServerError(
        err instanceof ApiError
          ? err.message
          : 'Something went wrong. Please try again.',
      )
    }
  }

  if (status === 'success') {
    return (
      <div className="contact-form contact-form--success" role="status">
        <h3>Thanks for reaching out!</h3>
        <p>
          Your message has been received. Your support ticket reference is{' '}
          <strong data-testid="ticket-reference">{ticketRef}</strong>. We&apos;ve sent a
          confirmation to your email and our team will reply soon.
        </p>
        <button
          type="button"
          onClick={() => {
            setStatus('idle')
            setTicketRef('')
          }}
        >
          Send another message
        </button>
      </div>
    )
  }

  const submitting = status === 'submitting'

  return (
    <form className="contact-form" onSubmit={handleSubmit} noValidate aria-label="Contact form">
      <h2>Contact us</h2>

      <div className="contact-form__field">
        <label htmlFor="contact-name">Name</label>
        <input
          id="contact-name"
          value={form.name}
          onChange={(e) => update('name', e.target.value)}
          aria-invalid={errors.name ? 'true' : undefined}
          disabled={submitting}
        />
        {errors.name && <p className="contact-form__error" role="alert">{errors.name}</p>}
      </div>

      <div className="contact-form__field">
        <label htmlFor="contact-email">Email</label>
        <input
          id="contact-email"
          type="email"
          value={form.email}
          onChange={(e) => update('email', e.target.value)}
          aria-invalid={errors.email ? 'true' : undefined}
          disabled={submitting}
        />
        {errors.email && <p className="contact-form__error" role="alert">{errors.email}</p>}
      </div>

      <div className="contact-form__field">
        <label htmlFor="contact-subject">Subject</label>
        <input
          id="contact-subject"
          value={form.subject}
          onChange={(e) => update('subject', e.target.value)}
          aria-invalid={errors.subject ? 'true' : undefined}
          disabled={submitting}
        />
        {errors.subject && <p className="contact-form__error" role="alert">{errors.subject}</p>}
      </div>

      <div className="contact-form__field">
        <label htmlFor="contact-message">Message</label>
        <textarea
          id="contact-message"
          rows={5}
          value={form.message}
          onChange={(e) => update('message', e.target.value)}
          aria-invalid={errors.message ? 'true' : undefined}
          disabled={submitting}
        />
        {errors.message && <p className="contact-form__error" role="alert">{errors.message}</p>}
      </div>

      {recaptchaSiteKey && (
        <div
          className="g-recaptcha"
          data-sitekey={recaptchaSiteKey}
          data-testid="recaptcha"
        />
      )}

      {serverError && (
        <p className="contact-form__error" role="alert">{serverError}</p>
      )}

      <button type="submit" disabled={submitting}>
        {submitting ? 'Sending…' : 'Send message'}
      </button>
    </form>
  )
}

export default ContactForm
