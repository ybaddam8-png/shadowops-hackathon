import React from 'react'
import { Clock } from 'lucide-react'
import { Scenario } from '../data/scenarios'
import { DomainBadge, RiskBadge, DecisionBadge } from './DecisionBadge'

interface ScenarioFeedProps {
  scenarios: Scenario[]
  selectedId: string
  onSelect: (id: string) => void
}

export const ScenarioFeed: React.FC<ScenarioFeedProps> = ({
  scenarios,
  selectedId,
  onSelect,
}) => {
  return (
    <div className="card p-0 overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-700 bg-slate-850">
        <h3 className="font-semibold text-slate-100">Scenario Feed</h3>
      </div>
      <div className="divide-y divide-slate-700">
        {scenarios.map((scenario) => (
          <button
            key={scenario.id}
            onClick={() => onSelect(scenario.id)}
            className={`w-full text-left p-4 transition-all duration-200 ${
              selectedId === scenario.id
                ? 'bg-slate-700 border-l-4 border-l-cyan-500'
                : 'hover:bg-slate-750'
            }`}
          >
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div className="font-semibold text-slate-100 flex-1">
                  {scenario.title}
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <DomainBadge domain={scenario.domain} />
                <RiskBadge level={scenario.riskLevel} />
                <DecisionBadge decision={scenario.decision} />
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Clock className="w-3 h-3" />
                <span>{new Date(scenario.timestamp).toLocaleTimeString()}</span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
