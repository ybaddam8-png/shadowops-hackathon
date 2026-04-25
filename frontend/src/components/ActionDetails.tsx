import React from 'react'
import { AlertTriangle } from 'lucide-react'
import { Scenario } from '../data/scenarios'

interface ActionDetailsProps {
  scenario: Scenario
}

export const ActionDetails: React.FC<ActionDetailsProps> = ({ scenario }) => {
  return (
    <div className="card space-y-6">
      <div>
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-2">
          Intent
        </h3>
        <p className="text-slate-100">{scenario.intent}</p>
      </div>

      <div>
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-2">
          Payload Summary
        </h3>
        <div className="bg-slate-900 rounded px-4 py-3 font-mono text-sm text-slate-300 break-words">
          {scenario.payload}
        </div>
      </div>

      <div>
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-2">
          Risk Vectors
        </h3>
        <div className="space-y-2">
          {scenario.riskVectors.map((vector, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
              <span className="text-slate-300">{vector}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="pt-2 border-t border-slate-700">
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-3">
          MITRE Mapping
        </h3>
        <div className="space-y-1">
          <div>
            <p className="text-xs text-slate-500">Tactic</p>
            <p className="text-slate-100 font-medium">{scenario.mireTactic}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Technique</p>
            <p className="text-slate-100 font-medium">{scenario.mireTechnique}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
