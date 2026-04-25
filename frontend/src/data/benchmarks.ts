export interface BenchmarkRow {
  policy: string
  exactMatch: number
  safetyAccuracy: number
  unsafeDecisionRate: number
  rewardMean: number
  isHighlight?: boolean
}

export const BENCHMARKS: BenchmarkRow[] = [
  {
    policy: 'Random',
    exactMatch: 50.0,
    safetyAccuracy: 68.8,
    unsafeDecisionRate: 31.2,
    rewardMean: 0.531,
  },
  {
    policy: 'Heuristic',
    exactMatch: 46.9,
    safetyAccuracy: 87.5,
    unsafeDecisionRate: 12.5,
    rewardMean: 0.891,
  },
  {
    policy: 'Q-Aware Supervisor',
    exactMatch: 78.1,
    safetyAccuracy: 100,
    unsafeDecisionRate: 0.0,
    rewardMean: 1.672,
    isHighlight: true,
  },
  {
    policy: 'Oracle',
    exactMatch: 100,
    safetyAccuracy: 100,
    unsafeDecisionRate: 0.0,
    rewardMean: 2.0,
  },
]

export const BENCHMARK_CARDS = [
  {
    title: 'Best Laptop-Safe Policy',
    value: 'Q-Aware Supervisor',
    status: 'verified' as const,
  },
  {
    title: 'GPU Training Status',
    value: 'Pending Cloud Run',
    status: 'pending' as const,
  },
  {
    title: 'Demo Readiness',
    value: 'Verified',
    status: 'verified' as const,
  },
]
