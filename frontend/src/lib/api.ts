// API stub for future backend integration
// For now, all functions return mock data

import { SCENARIOS, Scenario } from '../data/scenarios'
import { BENCHMARKS, BenchmarkRow } from '../data/benchmarks'

export interface HealthResponse {
  status: 'online' | 'offline'
  timestamp: string
}

export interface DecisionResponse {
  decision: 'ALLOW' | 'BLOCK' | 'QUARANTINE' | 'FORK'
  confidence: number
  reason: string
}

export interface BenchmarkResponse {
  benchmarks: BenchmarkRow[]
}

export async function getHealth(): Promise<HealthResponse> {
  return {
    status: 'online',
    timestamp: new Date().toISOString(),
  }
}

export async function simulateScenario(
  scenarioId: string
): Promise<Scenario | null> {
  const scenario = SCENARIOS.find((s) => s.id === scenarioId)
  return scenario || null
}

export async function getDecision(scenarioId: string): Promise<DecisionResponse> {
  const scenario = await simulateScenario(scenarioId)
  if (!scenario) {
    return {
      decision: 'QUARANTINE',
      confidence: 0,
      reason: 'Scenario not found',
    }
  }
  return {
    decision: scenario.decision,
    confidence: scenario.confidence,
    reason: scenario.reason,
  }
}

export async function getBenchmark(): Promise<BenchmarkResponse> {
  return {
    benchmarks: BENCHMARKS,
  }
}

export async function getDemoScenarios(): Promise<Scenario[]> {
  return SCENARIOS
}
