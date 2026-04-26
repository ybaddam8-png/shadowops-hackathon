import { SCENARIOS, Scenario } from '../data/scenarios'
import { BENCHMARKS, BenchmarkRow } from '../data/benchmarks'
const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000'

export interface HealthResponse {
  status: 'online' | 'offline'
  timestamp: string
}

export interface BackendStatusResponse {
  status: 'online' | 'offline'
  live: boolean
  ready: boolean
  mode: string
  modelLoaded: boolean
  classifierLoaded: boolean
  adapterLoaded: boolean
  startupError: string | null
  timestamp: string
  source: 'live' | 'fallback'
}

export interface DecisionResponse {
  decision: 'ALLOW' | 'BLOCK' | 'QUARANTINE' | 'FORK'
  confidence: number
  reason: string
  source: 'live' | 'fallback'
  modelMode: string
  uncertainty: number | null
  riskScore: number | null
  safeOutcome: string | null
  missingEvidence: string[]
  backendReady: boolean
}

export interface BenchmarkResponse {
  benchmarks: BenchmarkRow[]
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health`)
    if (!res.ok) throw new Error('health request failed')
    return { status: 'online', timestamp: new Date().toISOString() }
  } catch {
    return { status: 'offline', timestamp: new Date().toISOString() }
  }
}

export async function getBackendStatus(): Promise<BackendStatusResponse> {
  const timestamp = new Date().toISOString()
  try {
    const healthRes = await fetch(`${API_BASE}/health`)
    if (!healthRes.ok) throw new Error('health request failed')

    let readyPayload: any = null
    try {
      const readyRes = await fetch(`${API_BASE}/ready`)
      if (readyRes.ok) {
        readyPayload = await readyRes.json()
      }
    } catch {
      readyPayload = null
    }

    const ready = readyPayload?.status === 'ready'
    return {
      status: 'online',
      live: true,
      ready,
      mode: readyPayload?.mode || 'unknown',
      modelLoaded: Boolean(readyPayload?.model_loaded),
      classifierLoaded: Boolean(readyPayload?.classifier_loaded),
      adapterLoaded: Boolean(readyPayload?.adapter_loaded),
      startupError: (readyPayload?.startup_error as string | null) || null,
      timestamp,
      source: 'live',
    }
  } catch {
    return {
      status: 'offline',
      live: false,
      ready: false,
      mode: 'demo_fallback',
      modelLoaded: false,
      classifierLoaded: false,
      adapterLoaded: false,
      startupError: 'Backend unavailable. Frontend is using fallback demo data.',
      timestamp,
      source: 'fallback',
    }
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
      source: 'fallback',
      modelMode: 'demo_fallback',
      uncertainty: null,
      riskScore: null,
      safeOutcome: null,
      missingEvidence: [],
      backendReady: false,
    }
  }
  const domainMap: Record<string, string> = {
    github: 'GITHUB',
    aws: 'AWS',
    iam: 'AWS',
    network: 'SOC',
    pentest: 'SOC',
  }
  try {
    const payload = {
      domain: domainMap[scenario.domain] || 'SOC',
      action: {
        intent: scenario.intent,
        raw_payload: scenario.payload,
      },
      actor: 'frontend_ui',
      session_id: 'frontend-demo',
      service: scenario.domain,
      environment: 'production',
      provided_evidence: [],
    }
    const res = await fetch(`${API_BASE}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) throw new Error('decision request failed')
    const data = await res.json()
    const supervisor = data?.supervisor_decision || {}
    const confidenceRaw = supervisor.confidence
    const normalizedConfidence =
      typeof confidenceRaw === 'number'
        ? confidenceRaw <= 1
          ? confidenceRaw * 100
          : confidenceRaw
        : scenario.confidence
    return {
      decision: (supervisor.action_taken || scenario.decision) as DecisionResponse['decision'],
      confidence: Math.round(normalizedConfidence),
      reason: (supervisor.explanation || scenario.reason) as string,
      source: 'live',
      modelMode: (supervisor.policy_name || 'unknown') as string,
      uncertainty:
        typeof supervisor.uncertainty === 'number' ? Number(supervisor.uncertainty) : null,
      riskScore:
        typeof supervisor.cumulative_risk_score === 'number'
          ? Number(supervisor.cumulative_risk_score)
          : typeof supervisor.risk_score === 'number'
            ? Number(supervisor.risk_score)
            : null,
      safeOutcome:
        typeof supervisor.safe_outcome === 'string' ? supervisor.safe_outcome : null,
      missingEvidence: Array.isArray(supervisor.missing_evidence)
        ? supervisor.missing_evidence
        : [],
      backendReady: true,
    }
  } catch {
    return {
      decision: scenario.decision,
      confidence: scenario.confidence,
      reason: scenario.reason,
      source: 'fallback',
      modelMode: 'demo_fallback',
      uncertainty: null,
      riskScore: null,
      safeOutcome: null,
      missingEvidence: [],
      backendReady: false,
    }
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
