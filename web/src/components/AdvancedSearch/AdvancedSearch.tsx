import { useState, useEffect, useRef, useCallback } from 'react'

export interface AdvancedSearchFilters {
  query?: string
  startDate?: string
  endDate?: string
  assetType?: string
  minAmount?: number
  maxAmount?: number
}

export interface SearchSuggestion {
  label: string
  value: string
  type: 'account' | 'transaction' | 'recent'
}

const STORAGE_KEY_HISTORY = 'astroml:search:history'
const STORAGE_KEY_SAVED = 'astroml:search:saved'
const MAX_HISTORY = 10

function loadFromStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

function saveToStorage<T>(key: string, value: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // storage unavailable — no-op
  }
}

interface Props {
  onSearch: (filters: AdvancedSearchFilters) => void
  suggestions?: SearchSuggestion[]
  placeholder?: string
  debounceMs?: number
}

export function AdvancedSearch({
  onSearch,
  suggestions = [],
  placeholder = 'Search accounts, transactions…',
  debounceMs = 300,
}: Props) {
  const [query, setQuery] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [assetType, setAssetType] = useState('')
  const [minAmount, setMinAmount] = useState('')
  const [maxAmount, setMaxAmount] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [searchHistory, setSearchHistory] = useState<string[]>(() =>
    loadFromStorage<string[]>(STORAGE_KEY_HISTORY, [])
  )
  const [savedQueries, setSavedQueries] = useState<AdvancedSearchFilters[]>(() =>
    loadFromStorage<AdvancedSearchFilters[]>(STORAGE_KEY_SAVED, [])
  )
  const [showSaved, setShowSaved] = useState(false)

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const buildFilters = useCallback((): AdvancedSearchFilters => {
    const filters: AdvancedSearchFilters = {}
    if (query.trim()) filters.query = query.trim()
    if (startDate) filters.startDate = startDate
    if (endDate) filters.endDate = endDate
    if (assetType) filters.assetType = assetType
    if (minAmount !== '') filters.minAmount = Number(minAmount)
    if (maxAmount !== '') filters.maxAmount = Number(maxAmount)
    return filters
  }, [query, startDate, endDate, assetType, minAmount, maxAmount])

  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      onSearch(buildFilters())
    }, debounceMs)
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current)
    }
  }, [query, startDate, endDate, assetType, minAmount, maxAmount, buildFilters, onSearch, debounceMs])

  const commitSearch = () => {
    const q = query.trim()
    if (!q) return
    const updated = [q, ...searchHistory.filter((h) => h !== q)].slice(0, MAX_HISTORY)
    setSearchHistory(updated)
    saveToStorage(STORAGE_KEY_HISTORY, updated)
    setShowSuggestions(false)
  }

  const saveCurrentQuery = () => {
    const filters = buildFilters()
    if (Object.keys(filters).length === 0) return
    const updated = [filters, ...savedQueries.filter((s) => JSON.stringify(s) !== JSON.stringify(filters))].slice(0, 20)
    setSavedQueries(updated)
    saveToStorage(STORAGE_KEY_SAVED, updated)
  }

  const loadSavedQuery = (filters: AdvancedSearchFilters) => {
    setQuery(filters.query ?? '')
    setStartDate(filters.startDate ?? '')
    setEndDate(filters.endDate ?? '')
    setAssetType(filters.assetType ?? '')
    setMinAmount(filters.minAmount !== undefined ? String(filters.minAmount) : '')
    setMaxAmount(filters.maxAmount !== undefined ? String(filters.maxAmount) : '')
    setShowSaved(false)
  }

  const clearHistory = () => {
    setSearchHistory([])
    saveToStorage(STORAGE_KEY_HISTORY, [])
  }

  const allSuggestions: SearchSuggestion[] = [
    ...searchHistory.map((h): SearchSuggestion => ({ label: h, value: h, type: 'recent' })),
    ...suggestions.filter((s) => s.label.toLowerCase().includes(query.toLowerCase())),
  ]

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px 12px',
    border: '1px solid #ddd',
    borderRadius: 4,
    fontSize: 14,
    boxSizing: 'border-box',
  }

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: 4,
    fontSize: 13,
    fontWeight: 600,
    color: '#555',
  }

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {/* Full-text search row */}
      <div style={{ position: 'relative' }}>
        <label style={labelStyle}>Search</label>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setShowSuggestions(true)
              }}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitSearch()
                if (e.key === 'Escape') setShowSuggestions(false)
              }}
              placeholder={placeholder}
              style={inputStyle}
              aria-label="Search query"
              aria-autocomplete="list"
              aria-expanded={showSuggestions && allSuggestions.length > 0}
            />
            {showSuggestions && allSuggestions.length > 0 && (
              <ul
                role="listbox"
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  zIndex: 10,
                  margin: 0,
                  padding: 0,
                  listStyle: 'none',
                  background: '#fff',
                  border: '1px solid #ddd',
                  borderTop: 'none',
                  borderRadius: '0 0 4px 4px',
                  maxHeight: 240,
                  overflowY: 'auto',
                }}
              >
                {searchHistory.length > 0 && (
                  <li style={{ padding: '6px 12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f5f5f5' }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase' }}>Recent</span>
                    <button
                      onMouseDown={(e) => { e.preventDefault(); clearHistory() }}
                      style={{ border: 'none', background: 'none', color: '#888', cursor: 'pointer', fontSize: 11 }}
                    >
                      Clear
                    </button>
                  </li>
                )}
                {allSuggestions.map((s, i) => (
                  <li
                    key={i}
                    role="option"
                    aria-selected={false}
                    onMouseDown={() => {
                      setQuery(s.value)
                      commitSearch()
                      inputRef.current?.focus()
                    }}
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: 14,
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#f0f0f0' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = '' }}
                  >
                    <span style={{ fontSize: 10, color: '#aaa', textTransform: 'uppercase' }}>{s.type}</span>
                    <span>{s.label}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            onClick={saveCurrentQuery}
            title="Save this search"
            style={{
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: 4,
              background: '#fff',
              cursor: 'pointer',
              fontSize: 13,
              whiteSpace: 'nowrap',
            }}
          >
            Save
          </button>
          <button
            onClick={() => setShowSaved((v) => !v)}
            title="Saved searches"
            style={{
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: 4,
              background: showSaved ? '#f0f0f0' : '#fff',
              cursor: 'pointer',
              fontSize: 13,
              whiteSpace: 'nowrap',
            }}
          >
            Saved ({savedQueries.length})
          </button>
        </div>

        {showSaved && savedQueries.length > 0 && (
          <ul
            style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              zIndex: 10,
              margin: '4px 0 0',
              padding: 0,
              listStyle: 'none',
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: 4,
              minWidth: 260,
              maxHeight: 300,
              overflowY: 'auto',
            }}
          >
            {savedQueries.map((sq, i) => (
              <li
                key={i}
                onMouseDown={() => loadSavedQuery(sq)}
                style={{ padding: '8px 12px', cursor: 'pointer', fontSize: 13, borderBottom: '1px solid #f0f0f0' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#f5f5f5' }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = '' }}
              >
                {sq.query && <span style={{ fontWeight: 600 }}>{sq.query}</span>}
                {sq.startDate && <span style={{ color: '#888', marginLeft: 8, fontSize: 11 }}>from {sq.startDate}</span>}
                {sq.endDate && <span style={{ color: '#888', marginLeft: 4, fontSize: 11 }}>to {sq.endDate}</span>}
                {sq.assetType && <span style={{ color: '#888', marginLeft: 8, fontSize: 11 }}>{sq.assetType}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Filter row */}
      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}>
        <div>
          <label style={labelStyle}>Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            style={inputStyle}
            aria-label="Start date filter"
          />
        </div>

        <div>
          <label style={labelStyle}>End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            style={inputStyle}
            aria-label="End date filter"
          />
        </div>

        <div>
          <label style={labelStyle}>Asset Type</label>
          <select
            value={assetType}
            onChange={(e) => setAssetType(e.target.value)}
            style={inputStyle}
            aria-label="Asset type filter"
          >
            <option value="">All Assets</option>
            <option value="XLM">XLM</option>
            <option value="USDC">USDC</option>
            <option value="BTC">BTC</option>
            <option value="ETH">ETH</option>
          </select>
        </div>

        <div>
          <label style={labelStyle}>Min Amount</label>
          <input
            type="number"
            min="0"
            value={minAmount}
            onChange={(e) => setMinAmount(e.target.value)}
            placeholder="0"
            style={inputStyle}
            aria-label="Minimum amount filter"
          />
        </div>

        <div>
          <label style={labelStyle}>Max Amount</label>
          <input
            type="number"
            min="0"
            value={maxAmount}
            onChange={(e) => setMaxAmount(e.target.value)}
            placeholder="Any"
            style={inputStyle}
            aria-label="Maximum amount filter"
          />
        </div>
      </div>
    </div>
  )
}
