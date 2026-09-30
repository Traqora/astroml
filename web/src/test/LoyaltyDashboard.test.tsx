import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '../contexts/ThemeContext'
import App from '../App'

function renderWithProviders(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider>{ui}</ThemeProvider>
    </QueryClientProvider>
  )
}

test('renders dashboard sections', async () => {
  renderWithProviders(<App />)

  await waitFor(() => expect(screen.getByText(/AstroML Dashboard/i)).toBeInTheDocument())
  expect(screen.getByText(/🌙 Dark/i)).toBeInTheDocument()
})
