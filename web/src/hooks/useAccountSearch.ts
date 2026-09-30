import { useState, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { searchAccounts, getAutocompleteSuggestions } from '../api/search'
import type { SearchResult } from '../api/search'

const DEBOUNCE_MS = 300

export function useAccountSearch() {
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(-1)
  const [showResults, setShowResults] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setDebouncedQuery(query)
      setActiveIndex(-1)
    }, DEBOUNCE_MS)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [query])

  const { data: searchData, isLoading: searchLoading } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => searchAccounts(debouncedQuery, { limit: 10 }),
    enabled: debouncedQuery.length >= 2,
  })

  const { data: autocompleteData, isLoading: autocompleteLoading } = useQuery({
    queryKey: ['autocomplete', query],
    queryFn: () => getAutocompleteSuggestions(query),
    enabled: query.length >= 1 && query.length < 2,
  })

  const results: SearchResult[] = searchData?.results ?? []
  const suggestions: string[] = autocompleteData?.suggestions ?? []
  const isLoading = searchLoading || autocompleteLoading

  const totalResults = results.length + suggestions.length

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex((prev) => (prev + 1) % totalResults)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((prev) => (prev - 1 + totalResults) % totalResults)
    } else if (e.key === 'Escape') {
      setShowResults(false)
      setActiveIndex(-1)
    }
  }

  return {
    query,
    setQuery,
    results,
    suggestions,
    isLoading,
    showResults,
    setShowResults,
    activeIndex,
    setActiveIndex,
    handleKeyDown,
  }
}
