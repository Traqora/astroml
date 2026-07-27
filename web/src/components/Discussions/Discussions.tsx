import React, { useEffect, useState } from 'react'
import { get, post } from '../../api/client'
import './Discussions.css'

interface Discussion {
  id: string
  title: string
  body: string
  createdAt: string
  updatedAt: string
  author: string
  category: string
  commentCount: number
}

interface Category {
  name: string
  description: string
}

interface UserReputation {
  username: string
  reputationScore: number
  contributions: {
    commits: number
    issues: number
    pullRequests: number
  }
  followers: number
  repositories: number
}

export function Discussions() {
  const [discussions, setDiscussions] = useState<Discussion[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showNewDiscussionForm, setShowNewDiscussionForm] = useState(false)
  const [selectedAuthor, setSelectedAuthor] = useState<string | null>(null)
  const [userReputation, setUserReputation] = useState<UserReputation | null>(null)

  // Fetch categories on mount
  useEffect(() => {
    loadCategories()
    loadDiscussions()
  }, [])

  // Fetch discussions when category changes
  useEffect(() => {
    if (selectedCategory !== null) {
      loadDiscussions()
    }
  }, [selectedCategory])

  async function loadDiscussions() {
    setLoading(true)
    setError(null)
    try {
      const endpoint = `/api/v1/discussions/recent?category=${selectedCategory || ''}&limit=50`
      const data = await get<{ discussions: Discussion[] }>(endpoint)
      setDiscussions(data.discussions)
    } catch (err) {
      setError('Failed to load discussions')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  async function loadCategories() {
    try {
      const data = await get<{ categories: Category[] }>('/api/v1/discussions/categories')
      setCategories(data.categories)
    } catch (err) {
      console.error('Failed to load categories:', err)
    }
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!searchQuery.trim()) return

    setLoading(true)
    setError(null)
    try {
      const data = await post<{ results: Discussion[] }>('/api/v1/discussions/search', {
        query: searchQuery,
        limit: 50,
      })
      setDiscussions(data.results)
    } catch (err) {
      setError('Search failed')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  async function handleAuthorClick(author: string) {
    setSelectedAuthor(author)
    setLoading(true)
    try {
      const data = await get<UserReputation>(`/api/v1/discussions/user-reputation/${author}`)
      setUserReputation(data)
    } catch (err) {
      console.error('Failed to fetch user reputation:', err)
    } finally {
      setLoading(false)
    }
  }

  function closeReputationModal() {
    setSelectedAuthor(null)
    setUserReputation(null)
  }

  const filteredDiscussions = selectedCategory
    ? discussions.filter(d => d.category === selectedCategory)
    : discussions

  return (
    <div className="discussions-container">
      <header className="discussions-header">
        <h1>Community Discussions</h1>
        <p>Join the conversation and share your insights</p>
      </header>

      <div className="discussions-controls">
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="Search discussions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="search-button">Search</button>
        </form>

        <button
          className="new-discussion-btn"
          onClick={() => setShowNewDiscussionForm(!showNewDiscussionForm)}
        >
          + New Discussion
        </button>
      </div>

      {showNewDiscussionForm && (
        <div className="new-discussion-form">
          <h3>Start a New Discussion</h3>
          <p>Create a new discussion on GitHub</p>
          <a
            href={`https://github.com/Traqora/astroml/discussions/new`}
            target="_blank"
            rel="noopener noreferrer"
            className="github-link-btn"
          >
            Open GitHub Discussions →
          </a>
        </div>
      )}

      <div className="discussions-filters">
        <button
          className={`filter-btn ${selectedCategory === null ? 'active' : ''}`}
          onClick={() => setSelectedCategory(null)}
        >
          All
        </button>
        {categories.map((cat) => (
          <button
            key={cat.name}
            className={`filter-btn ${selectedCategory === cat.name ? 'active' : ''}`}
            onClick={() => setSelectedCategory(cat.name)}
            title={cat.description}
          >
            {cat.name}
          </button>
        ))}
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {loading ? (
        <div className="loading-state">Loading discussions...</div>
      ) : filteredDiscussions.length === 0 ? (
        <div className="empty-state">
          <p>No discussions found</p>
        </div>
      ) : (
        <div className="discussions-list">
          {filteredDiscussions.map((discussion) => (
            <div key={discussion.id} className="discussion-card">
              <div className="discussion-header">
                <h3>
                  <a
                    href={`https://github.com/Traqora/astroml/discussions/${discussion.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {discussion.title}
                  </a>
                </h3>
                <span className="category-badge">{discussion.category}</span>
              </div>

              <p className="discussion-body">{discussion.body}</p>

              <div className="discussion-meta">
                <span className="author" onClick={() => handleAuthorClick(discussion.author)}>
                  By <strong>{discussion.author}</strong>
                </span>
                <span className="comments">
                  💬 {discussion.commentCount} comment{discussion.commentCount !== 1 ? 's' : ''}
                </span>
                <span className="date">
                  {new Date(discussion.updatedAt).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedAuthor && userReputation && (
        <div className="reputation-modal" onClick={closeReputationModal}>
          <div className="reputation-card" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={closeReputationModal}>×</button>
            
            <h3>{userReputation.username}</h3>
            
            <div className="reputation-score">
              <div className="score-value">{userReputation.reputationScore}</div>
              <div className="score-label">Reputation Score</div>
            </div>

            <div className="reputation-stats">
              <div className="stat">
                <div className="stat-value">{userReputation.contributions.commits}</div>
                <div className="stat-label">Commits</div>
              </div>
              <div className="stat">
                <div className="stat-value">{userReputation.contributions.issues}</div>
                <div className="stat-label">Issues</div>
              </div>
              <div className="stat">
                <div className="stat-value">{userReputation.contributions.pullRequests}</div>
                <div className="stat-label">Pull Requests</div>
              </div>
              <div className="stat">
                <div className="stat-value">{userReputation.followers}</div>
                <div className="stat-label">Followers</div>
              </div>
              <div className="stat">
                <div className="stat-value">{userReputation.repositories}</div>
                <div className="stat-label">Repositories</div>
              </div>
            </div>

            <a
              href={`https://github.com/${userReputation.username}`}
              target="_blank"
              rel="noopener noreferrer"
              className="github-profile-link"
            >
              View GitHub Profile →
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

export default Discussions
