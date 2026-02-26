Loyalty Dashboard – Technical Documentation

Overview

A React + TypeScript dashboard for customer loyalty: shows current tier, points balance, tier benefits, earning history with pagination, progress to next tier, points redemption, tier comparison chart, and referral invites.
Visualizations use Recharts. Points redemption uses optimistic updates. Confetti animation triggers on tier upgrades.
Location in repository

web/
src/App.tsx, src/main.tsx
src/lib/types.ts, src/lib/confetti.ts
src/api/loyalty.ts (mock API)
src/hooks/useLoyaltySummary.ts, usePointsHistory.ts, useRedeemPoints.ts
src/components/LoyaltyDashboard/* (all UI components)
src/test/* (tests)
Tooling: Vite (vite.config.ts), Vitest (vitest.setup.ts), TypeScript (tsconfig.json), package (package.json), index.html
Primary features and acceptance criteria mapping

Loyalty dashboard page: src/components/LoyaltyDashboard/LoyaltyDashboard.tsx
Tier progress visualization: src/components/LoyaltyDashboard/TierProgress.tsx
Points history table with pagination: src/components/LoyaltyDashboard/PointsHistoryTable.tsx
Tier benefits display: src/components/LoyaltyDashboard/TierBenefitsCard.tsx
Points redemption UI (optimistic updates): src/components/LoyaltyDashboard/PointsRedemptionPanel.tsx, src/hooks/useRedeemPoints.ts
Tier comparison chart: src/components/LoyaltyDashboard/TierComparisonChart.tsx
Referral invitation section: src/components/LoyaltyDashboard/ReferralInviteSection.tsx
Tests: src/test/*.test.tsx
Tech stack

React 18 + TypeScript
Data: @tanstack/react-query for fetching, caching, optimistic mutations
Charts: recharts (RadialBarChart, BarChart)
Confetti: canvas-confetti (wrapped in lib/confetti.ts)
Testing: Vitest + Testing Library + jsdom
Mock API: In-memory implementation in src/api/loyalty.ts
Data contracts (conceptual)

LoyaltyTier: id, name, threshold, multiplier, color
Benefit: id, title, description, icon?
LoyaltySummary: currentTier, pointsBalance, nextTier?, benefits[]
nextTier: tier, remainingToUpgrade, progressPct (0..100)
PointsTransaction: id, date (ISO), type (earn|redeem|adjust), points, source?, note?
PointsHistoryResponse: data[], page, pageSize, total
RedemptionRequest: points, rewardId?
RedemptionResponse: newBalance, transaction
TierComparisonDatum: tier, threshold, multiplier, retention
API layer (mock, self-contained)

getLoyaltySummary(): computes next tier progress and benefits; returns LoyaltySummary
getPointsHistory(page, pageSize): paginated list of transactions
redeemPoints(request): validates and records redemption, returns updated balance and transaction
getTierComparison(): static comparison dataset
getReferralLink(): returns URL and invite stats
Replace with real backend by preserving function signatures and swapping implementations.
UI components

LoyaltyDashboard
Orchestrates data loading, pagination state, and renders sections in order
TierProgress
Radial progress chart (Recharts) showing nextTier.progressPct
Displays remaining points to next tier or “Top tier achieved”
Confetti triggers when currentTier.id changes between renders (via useRef + fireConfetti)
PointsHistoryTable
Tabular display: Date, Type, Points, Source, Note
Prev/Next pagination, responsive table
keepPreviousData enabled for smooth page transitions
TierBenefitsCard
Grid of benefit tiles with title and description
PointsRedemptionPanel
Numeric input with basic validation (positive, <= balance)
Calls onRedeem; pending state disables input
TierComparisonChart
Grouped BarChart (threshold, multiplier, retention) with tooltip/legend
ReferralInviteSection
Displays referral URL, copy-to-clipboard, Web Share API fallback
Shows invited/rewards counts
Optimistic redemption updates

Implemented in useRedeemPoints:
onMutate: cancel in-flight queries; snapshot cache; optimistically decrement pointsBalance; prepend pending redemption transaction
onError: rollback to snapshots
onSuccess: reconcile with server (final balance and transaction)
onSettled: invalidate queries for fresh data
Confetti behavior

Triggered in TierProgress when currentTier.id changes compared to prior render
Uses a small wrapper (lib/confetti.ts) over canvas-confetti
Pagination

Page state managed in LoyaltyDashboard
Page size set to 10 (constant in the component)
keepPreviousData prevents content flash on page transitions
Next/Prev disabled appropriately at bounds or while loading
Testing

src/test/LoyaltyDashboard.test.tsx: verifies all sections render
src/test/TierProgress.test.tsx: checks percent/remaining display, confetti on tier change
src/test/PointsHistoryTable.test.tsx: validates rows render
src/test/PointsRedemptionPanel.test.tsx: validates input and that redeem callback is invoked
Tests run with Vitest in jsdom, RTL assertions via jest-dom
Setup and running

cd web
npm install
npm run dev (open printed local URL)
npm run test (execute unit tests)
Performance and UX considerations

Binary-cached data via React Query minimizes refetching
Optimistic path keeps interactions responsive
Recharts components wrapped in ResponsiveContainer for fluid layout
Table uses lightweight plain HTML for performance and accessibility
Accessibility

Semantic HTML in tables, buttons, inputs
Keyboard accessible pagination and actions
Informative labels and concise messaging
Extensibility

Swap mock API with real endpoints in src/api/loyalty.ts
Add sorting/filtering to PointsHistoryTable by extending query and UI controls
Add reward catalog selection to PointsRedemptionPanel
Internationalize labels and dates using i18n libraries
Add skeletons and toasts for richer feedback
Limitations and next steps

In-memory API is non-persistent; wire to a backend service
Add more comprehensive tests: error states, pagination requests sequencing, accessibility checks
Enhance tier progress with additional analytics (e.g., projected upgrade date)
This document describes the implementation as delivered in web/, with clear pointers to each module and how they satisfy the acceptance criteria and technical notes.

