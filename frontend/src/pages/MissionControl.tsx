import React, { useState } from 'react'
import { SCENARIOS } from '../data/scenarios'
import { ScenarioFeed } from '../components/ScenarioFeed'
import { ActionDetails } from '../components/ActionDetails'
import { SupervisorDecision } from '../components/SupervisorDecision'

export const MissionControl: React.FC<{ onNavigate?: (page: string) => void }> = () => {
  const [selectedId, setSelectedId] = useState('github-poisoned-pr')
  const selectedScenario = SCENARIOS.find((s) => s.id === selectedId) || SCENARIOS[0]

  const handleReplay = () => {
    setSelectedId(selectedId)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Mission Control Dashboard</h2>
        <p className="text-slate-400 mt-1">Manage incoming actions and supervisor decisions</p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - Scenario Feed */}
        <div>
          <ScenarioFeed
            scenarios={SCENARIOS}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>

        {/* Center Column - Action Details */}
        <div>
          <ActionDetails scenario={selectedScenario} />
        </div>

        {/* Right Column - Supervisor Decision */}
        <div>
          <SupervisorDecision
            scenario={selectedScenario}
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
