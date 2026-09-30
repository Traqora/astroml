import { get } from './client'

export interface SearchResult {
  id: string
  title: string
  content: string
  type: string
  score: number
  method: string
  metadata: Record<string, unknown>
}

export interface SearchResponse {
  query: string
  mode: string
  results: SearchResult[]
}

export interface AutocompleteResponse {
  suggestions: string[]
}

export async function searchAccounts(
  query: string,
  options?: { mode?: string; type?: string; limit?: number }
): Promise<SearchResponse> {
  const params = new URLSearchParams({ query })
  if (options?.mode) params.append('mode', options.mode)
  if (options?.type) params.append('type', options.type)
  if (options?.limit) params.append('limit', String(options.limit))

  const response = await get<SearchResponse>(`/api/v1/search?${params.toString()}`)
  return response
}

export async function getAutocompleteSuggestions(q: string): Promise<AutocompleteResponse> {
  const params = new URLSearchParams({ q })
  const response = await get<AutocompleteResponse>(`/api/v1/search/autocomplete?${params.toString()}`)
  return response
}
