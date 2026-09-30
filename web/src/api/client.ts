/**
 * API Client Module
 * 
 * Provides HTTP client with base URL configuration, auth token management,
 * error handling, and retry logic for the AstroML frontend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * Storage key for JWT token
 */
const TOKEN_KEY = 'astroml_auth_token'

/**
 * Get the current auth token from localStorage
 */
export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

/**
 * Set the auth token in localStorage
 */
export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

/**
 * Clear the auth token from localStorage
 */
export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

/**
 * API Error class for handling HTTP errors
 */
export class ApiError extends Error {
  status: number
  data: any

  constructor(status: number, message: string, data?: any) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
  }
}

/**
 * Request options interface
 */
interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  body?: any
  headers?: Record<string, string>
  retries?: number
  signal?: AbortSignal
}

/**
 * Make an HTTP request to the API
 * 
 * @param endpoint - API endpoint path (e.g., '/api/v1/transactions')
 * @param options - Request options
 * @returns Promise with response data
 * @throws ApiError on HTTP errors
 */
export async function apiRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const {
    method = 'GET',
    body,
    headers = {},
    retries = 3,
    signal,
  } = options

  const url = `${API_BASE_URL}${endpoint}`
  const token = getAuthToken()

  const requestHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    ...headers,
  }

  if (token) {
    requestHeaders['Authorization'] = `Bearer ${token}`
  }

  const config: RequestInit = {
    method,
    headers: requestHeaders,
    signal,
  }

  if (body) {
    config.body = JSON.stringify(body)
  }

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, config)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new ApiError(
          response.status,
          errorData.detail || errorData.message || `HTTP ${response.status}`,
          errorData
        )
      }

      // Handle 204 No Content
      if (response.status === 204) {
        return undefined as T
      }

      return await response.json()
    } catch (error) {
      lastError = error as Error

      // Don't retry on abort or 4xx errors (except 408, 429)
      if (error instanceof ApiError) {
        if (error.status >= 400 && error.status < 500 && error.status !== 408 && error.status !== 429) {
          throw error
        }
      }

      // Don't retry if this was the last attempt
      if (attempt === retries) {
        throw lastError
      }

      // Exponential backoff
      const delay = Math.min(1000 * Math.pow(2, attempt), 10000)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  throw lastError
}

/**
 * GET request helper
 */
export async function get<T>(endpoint: string, options?: Omit<RequestOptions, 'method'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'GET' })
}

/**
 * POST request helper
 */
export async function post<T>(endpoint: string, body?: any, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'POST', body })
}

/**
 * PUT request helper
 */
export async function put<T>(endpoint: string, body?: any, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'PUT', body })
}

/**
 * DELETE request helper
 */
export async function del<T>(endpoint: string, options?: Omit<RequestOptions, 'method'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'DELETE' })
}

/**
 * PATCH request helper
 */
export async function patch<T>(endpoint: string, body?: any, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'PATCH', body })
}
