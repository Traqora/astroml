import { useEffect, useState } from "react";

interface Badge {
  id: string;
  label: string;
  description: string;
}

interface Contributor {
  username: string;
  avatar_url: string;
  profile_url: string;
  commits: number;
  pull_requests: number;
  issues: number;
  total_contributions: number;
  badges: Badge[];
}

interface ActivityPoint {
  date: string;
  commits: number;
  pull_requests: number;
  issues: number;
}

type SortKey = "total" | "commits" | "pull_requests" | "issues";

interface Props {
  apiBase?: string;
  limit?: number;
}

export function ContributorsDashboard({ apiBase = "/api/v1", limit = 20 }: Props) {
  const [contributors, setContributors] = useState<Contributor[]>([]);
  const [newContributors, setNewContributors] = useState<Contributor[]>([]);
  const [activity, setActivity] = useState<ActivityPoint[]>([]);
  const [sortBy, setSortBy] = useState<SortKey>("total");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetch(`${apiBase}/contributors?limit=${limit}&sort_by=${sortBy}`).then((r) => {
        if (!r.ok) throw new Error(`contributors fetch failed: ${r.status}`);
        return r.json();
      }),
      fetch(`${apiBase}/contributors/new?days=30`).then((r) => {
        if (!r.ok) throw new Error(`new contributors fetch failed: ${r.status}`);
        return r.json();
      }),
      fetch(`${apiBase}/contributors/activity?days=30`).then((r) => {
        if (!r.ok) throw new Error(`activity fetch failed: ${r.status}`);
        return r.json();
      }),
    ])
      .then(([top, newC, act]) => {
        setContributors(top.contributors ?? []);
        setNewContributors(newC.contributors ?? []);
        setActivity(act.activity ?? []);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [apiBase, limit, sortBy]);

  if (loading) {
    return <div className="contributors-loading">Loading contributors…</div>;
  }

  if (error) {
    return <div className="contributors-error">Failed to load contributors: {error}</div>;
  }

  return (
    <div className="contributors-dashboard">
      <h2 className="contributors-title">Contributors</h2>

      {/* Sort controls */}
      <div className="contributors-sort">
        <span>Sort by:</span>
        {(["total", "commits", "pull_requests", "issues"] as SortKey[]).map((key) => (
          <button
            key={key}
            className={`sort-btn ${sortBy === key ? "active" : ""}`}
            onClick={() => setSortBy(key)}
          >
            {key === "pull_requests" ? "PRs" : key.charAt(0).toUpperCase() + key.slice(1)}
          </button>
        ))}
      </div>

      {/* New contributors highlight */}
      {newContributors.length > 0 && (
        <section className="new-contributors">
          <h3>New This Month</h3>
          <div className="contributor-chips">
            {newContributors.map((c) => (
              <a
                key={c.username}
                href={c.profile_url}
                target="_blank"
                rel="noopener noreferrer"
                className="contributor-chip"
                title={`${c.username} — ${c.commits} commits`}
              >
                <img src={c.avatar_url} alt={c.username} className="avatar-sm" />
                <span>{c.username}</span>
              </a>
            ))}
          </div>
        </section>
      )}

      {/* Activity sparkline (simple bar chart) */}
      {activity.length > 0 && (
        <section className="contribution-activity">
          <h3>Activity (last 30 days)</h3>
          <div className="activity-bars" aria-label="contribution activity chart">
            {activity.map((point) => {
              const maxCommits = Math.max(...activity.map((p) => p.commits), 1);
              const height = Math.round((point.commits / maxCommits) * 40) + 4;
              return (
                <div
                  key={point.date}
                  className="activity-bar"
                  style={{ height: `${height}px` }}
                  title={`${point.date}: ${point.commits} commits`}
                />
              );
            })}
          </div>
        </section>
      )}

      {/* Top contributors table */}
      <section className="top-contributors">
        <h3>Top Contributors</h3>
        {contributors.length === 0 ? (
          <p>No contributors found.</p>
        ) : (
          <ul className="contributor-list">
            {contributors.map((c, idx) => (
              <li key={c.username} className="contributor-row">
                <span className="rank">#{idx + 1}</span>
                <a
                  href={c.profile_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="contributor-identity"
                >
                  <img src={c.avatar_url} alt={c.username} className="avatar" />
                  <span className="username">{c.username}</span>
                </a>
                <div className="contributor-stats">
                  <span title="Commits">💾 {c.commits}</span>
                  <span title="Pull Requests">🔀 {c.pull_requests}</span>
                  <span title="Issues">🐛 {c.issues}</span>
                </div>
                {c.badges.length > 0 && (
                  <div className="contributor-badges">
                    {c.badges.map((b) => (
                      <span key={b.id} className="badge" title={b.description}>
                        {b.label}
                      </span>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
