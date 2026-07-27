import { useEffect, useState } from "react";

interface ChecklistItem {
  step: string;
  label: string;
  completed: boolean;
}

interface OnboardingProgress {
  github_username: string;
  checklist: ChecklistItem[];
  completed_count: number;
  total_steps: number;
  progress_pct: number;
  is_complete: boolean;
  started_at: string;
  last_updated: string;
}

interface Props {
  githubUsername: string;
  apiBase?: string;
}

export function OnboardingChecklist({ githubUsername, apiBase = "/api/v1" }: Props) {
  const [progress, setProgress] = useState<OnboardingProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [completing, setCompleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/contributors/onboarding/${githubUsername}`)
      .then((r) => r.json())
      .then(setProgress)
      .catch(() => setError("Failed to load onboarding progress"))
      .finally(() => setLoading(false));
  }, [githubUsername, apiBase]);

  async function markComplete(step: string) {
    setCompleting(step);
    try {
      const res = await fetch(
        `${apiBase}/contributors/onboarding/${githubUsername}/complete`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ step }),
        }
      );
      if (!res.ok) throw new Error("Request failed");
      setProgress(await res.json());
    } catch {
      setError("Failed to update step");
    } finally {
      setCompleting(null);
    }
  }

  if (loading) return <p className="onboarding-loading">Loading onboarding checklist…</p>;
  if (error) return <p className="onboarding-error">{error}</p>;
  if (!progress) return null;

  return (
    <div className="onboarding-checklist">
      <h2>Contributor Onboarding</h2>
      <p className="onboarding-username">@{progress.github_username}</p>

      <div className="onboarding-progress-bar">
        <div
          className="onboarding-progress-fill"
          style={{ width: `${progress.progress_pct}%` }}
        />
      </div>
      <p className="onboarding-progress-label">
        {progress.completed_count} / {progress.total_steps} steps complete ({progress.progress_pct}%)
      </p>

      {progress.is_complete && (
        <p className="onboarding-complete-badge">Onboarding complete!</p>
      )}

      <ul className="onboarding-steps">
        {progress.checklist.map((item) => (
          <li key={item.step} className={`onboarding-step ${item.completed ? "completed" : ""}`}>
            <span className="onboarding-step-icon">{item.completed ? "✓" : "○"}</span>
            <span className="onboarding-step-label">{item.label}</span>
            {!item.completed && (
              <button
                className="onboarding-step-btn"
                disabled={completing === item.step}
                onClick={() => markComplete(item.step)}
              >
                {completing === item.step ? "Saving…" : "Mark done"}
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
