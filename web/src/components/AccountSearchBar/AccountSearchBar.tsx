import { useRef, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useAccountSearch } from '../../hooks/useAccountSearch'

interface Props {
  onSelectResult?: (result: { id: string; type: string; title: string }) => void
}

export function AccountSearchBar({ onSelectResult }: Props) {
  const { t } = useTranslation()
  const {
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
  } = useAccountSearch()

  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)

  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const items = listRef.current.querySelectorAll('[role="option"]')
      items[activeIndex]?.scrollIntoView({ block: 'nearest' })
    }
  }, [activeIndex])

  const handleResultClick = (result: { id: string; type: string; title: string }) => {
    setShowResults(false)
    onSelectResult?.(result)
  }

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion)
    setShowResults(false)
  }

  const totalItems = results.length + suggestions.length

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ display: 'flex', gap: 8 }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setShowResults(true)
            }}
            onFocus={() => setShowResults(true)}
            onBlur={() => setTimeout(() => setShowResults(false), 150)}
            onKeyDown={handleKeyDown}
            placeholder={t('search.placeholder')}
            style={{
              width: '100%',
              padding: '10px 14px',
              border: '1px solid var(--border-color, #ddd)',
              borderRadius: 6,
              fontSize: 14,
              boxSizing: 'border-box',
              background: 'var(--bg-primary, #fff)',
              color: 'var(--text-primary, #1a202c)',
            }}
            aria-label="Search accounts and transactions"
            aria-autocomplete="list"
            aria-expanded={showResults && totalItems > 0}
            role="combobox"
          />

          {isLoading && query.length >= 1 && (
            <div style={{
              position: 'absolute',
              right: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              fontSize: 12,
              color: 'var(--text-muted, #999)',
            }}>
              {t('search.loading')}
            </div>
          )}
        </div>
      </div>

      {showResults && totalItems > 0 && (
        <ul
          ref={listRef}
          role="listbox"
          aria-label="Search results"
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            zIndex: 20,
            margin: '4px 0 0',
            padding: 0,
            listStyle: 'none',
            background: 'var(--bg-card, #fff)',
            border: '1px solid var(--border-color, #ddd)',
            borderRadius: 6,
            maxHeight: 320,
            overflowY: 'auto',
            boxShadow: 'var(--shadow-md, 0 4px 16px rgba(0,0,0,0.1))',
          }}
        >
          {suggestions.length > 0 && (
            <>
              <li style={{ padding: '6px 12px', fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #888)', textTransform: 'uppercase', background: 'var(--bg-secondary, #f5f5f5)' }}>
                {t('search.recent')}
              </li>
              {suggestions.map((s, i) => (
                <li
                  key={`suggestion-${i}`}
                  role="option"
                  aria-selected={activeIndex === i}
                  onMouseDown={() => handleSuggestionClick(s)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    fontSize: 14,
                    background: activeIndex === i ? 'var(--bg-secondary, #f0f0f0)' : 'transparent',
                  }}
                >
                  {s}
                </li>
              ))}
            </>
          )}

          {results.length > 0 && (
            <>
              <li style={{ padding: '6px 12px', fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #888)', textTransform: 'uppercase', background: 'var(--bg-secondary, #f5f5f5)' }}>
                {t('search.accounts')} & {t('search.transactions')}
              </li>
              {results.map((result, i) => {
                const itemIndex = suggestions.length + i
                return (
                  <li
                    key={result.id}
                    role="option"
                    aria-selected={activeIndex === itemIndex}
                    onMouseDown={() => handleResultClick(result)}
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: 14,
                      background: activeIndex === itemIndex ? 'var(--bg-secondary, #f0f0f0)' : 'transparent',
                      borderBottom: '1px solid var(--border-light, #f1f1f1)',
                    }}
                  >
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      color: result.type === 'account' ? '#0066cc' : '#28a745',
                      minWidth: 60,
                    }}>
                      {result.type}
                    </span>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {result.title}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted, #999)' }}>
                      {(result.score * 100).toFixed(0)}%
                    </span>
                  </li>
                )
              })}
            </>
          )}
        </ul>
      )}

      {showResults && query.length >= 2 && !isLoading && totalItems === 0 && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          zIndex: 20,
          margin: '4px 0 0',
          padding: '12px 16px',
          background: 'var(--bg-card, #fff)',
          border: '1px solid var(--border-color, #ddd)',
          borderRadius: 6,
          textAlign: 'center',
          color: 'var(--text-muted, #999)',
          fontSize: 13,
        }}>
          {t('search.no_results')}
        </div>
      )}
    </div>
  )
}
