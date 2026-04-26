import React, { useEffect, useState } from 'react'
import { SCENARIOS } from '../data/scenarios'
import { ScenarioFeed } from '../components/ScenarioFeed'
import { ActionDetails } from '../components/ActionDetails'
import { SupervisorDecision } from '../components/SupervisorDecision'
import { BackendStatusResponse, DecisionResponse, getBackendStatus, getDecision } from '../lib/api'

const DEFAULT_BACKEND_STATUS: BackendStatusResponse = {
  status: 'offline',
  live: false,
  ready: false,
  mode: 'demo_fallback',
  modelLoaded: false,
  classifierLoaded: false,
  adapterLoaded: false,
  startupError: 'Backend status not checked yet.',
  timestamp: new Date().toISOString(),
  source: 'fallback',
}

export const MissionControl: React.FC<{ onNavigate?: (page: string) => void }> = () => {
  const [selectedId, setSelectedId] = useState('github-poisoned-pr')
  const [backendStatus, setBackendStatus] = useState<BackendStatusResponse>(DEFAULT_BACKEND_STATUS)
  const [decisionByScenario, setDecisionByScenario] = useState<Record<string, DecisionResponse>>({})
  const [isRunningLiveDecision, setIsRunningLiveDecision] = useState(false)

  useEffect(() => {
    let active = true
    const refresh = async () => {
      const status = await getBackendStatus()
      if (active) setBackendStatus(status)
    }
    refresh()
    const intervalId = window.setInterval(refresh, 10000)
    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  const scenariosWithDecisions = SCENARIOS.map((scenario) => {
    const liveDecision = decisionByScenario[scenario.id]
    if (!liveDecision) return scenario
    return {
      ...scenario,
      decision: liveDecision.decision,
      confidence: liveDecision.confidence,
      reason: liveDecision.reason,
    }
  })

  const selectedScenario = scenariosWithDecisions.find((s) => s.id === selectedId) || scenariosWithDecisions[0]
  const selectedLiveDecision = decisionByScenario[selectedId] || null

  const handleRunLiveDecision = async () => {
    setIsRunningLiveDecision(true)
    const liveDecision = await getDecision(selectedId)
    setDecisionByScenario((prev) => ({
      ...prev,
      [selectedId]: liveDecision,
    }))
    setIsRunningLiveDecision(false)
  }

  const handleReplay = async () => {
    await handleRunLiveDecision()
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Mission Control Dashboard</h2>
        <p className="text-slate-400 mt-1">Manage incoming actions and supervisor decisions</p>
      </div>

      <div className={`border rounded-lg px-4 py-3 text-sm ${backendStatus.source === 'live' ? 'border-green-800 bg-green-900/20 text-green-200' : 'border-amber-800 bg-amber-900/20 text-amber-200'}`}>
        <div className="flex items-center justify-between gap-4">
          <div>
            <span className="font-semibold">
              {backendStatus.source === 'live' ? 'Live backend connected' : 'Fallback mode active'}
            </span>
            <span className="ml-2 text-xs opacity-80">
              mode={backendStatus.mode} | ready={String(backendStatus.ready)}
            </span>
          </div>
          <button
            onClick={handleRunLiveDecision}
            disabled={isRunningLiveDecision}
            className="btn btn-secondary"
          >
            {isRunningLiveDecision ? 'Running...' : 'Run Decision for Selected Scenario'}
          </button>
        </div>
        {backendStatus.startupError && backendStatus.source !== 'live' && (
          <p className="text-xs mt-2 opacity-90">{backendStatus.startupError}</p>
        )}
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div>
          <ScenarioFeed
            scenarios={scenariosWithDecisions}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>

        <div>
          <ActionDetails scenario={selectedScenario} />
        </div>

        <div>
          <SupervisorDecision
            scenario={selectedScenario}
            liveDecision={selectedLiveDecision}
            backendStatus={backendStatus}
            isLoading={isRunningLiveDecision}
            onRunLive={handleRunLiveDecision}
            onReplay={handleReplay}
            onApprove={() => console.log('Approved')}
            onBlock={() => console.log('Blocked')}
            onFork={() => console.log('Forked to human')}
          />
        </div>
      </div>
    </div>
  )
}
