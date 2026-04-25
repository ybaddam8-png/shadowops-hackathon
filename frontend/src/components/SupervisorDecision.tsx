import React from 'react'
import { Scenario } from '../data/scenarios'
import { DecisionBadge } from './DecisionBadge'

interface SupervisorDecisionProps {
  scenario: Scenario
  onApprove?: () => void
  onBlock?: () => void
  onFork?: () => void
  onReplay?: () => void
}

export const SupervisorDecision: React.FC<SupervisorDecisionProps> = ({
  scenario,
  onApprove,
  onBlock,
  onFork,
  onReplay,
}) => {
  return (
    <div className="card space-y-6">
      <div>
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-3">
          Supervisor Decision
        </h3>
        <div className="flex items-center gap-3">
          <DecisionBadge decision={scenario.decision} confidence={scenario.confidence} />
        </div>
      </div>

      <div className="bg-slate-900 rounded p-4 space-y-3">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
            Confidence
          </p>
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500 rounded-full"
                style={{ width: `${scenario.confidence}%` }}
              />
            </div>
            <span className="text-cyan-400 font-mono font-semibold">
              {scenario.confidence}%
            </span>
          </div>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-2">
          Reasoning
        </p>
        <p className="text-slate-300 leading-relaxed text-sm">{scenario.reason}</p>
      </div>

      {scenario.evidence?.status === 'verified' && (
        <div className="bg-green-900/20 border border-green-800 rounded-lg p-4">
          <p className="text-xs text-green-400 uppercase tracking-wide font-semibold mb-2">
            Evidence Verified
          </p>
          <p className="text-green-300 text-sm mb-3">{scenario.evidence.result}</p>
          {scenario.evidence.finalDecision && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-green-400">Final Decision:</span>
              <DecisionBadge decision={scenario.evidence.finalDecision} />
            </div>
          )}
          {scenario.evidence.restrictions && (
            <p className="text-xs text-green-300 mt-2">
              <span className="font-semibold">Restrictions:</span>{' '}
              {scenario.evidence.restrictions}
            </p>
          )}
        </div>
      )}

      <div className="pt-4 border-t border-slate-700 space-y-2">
        <button
          onClick={onApprove}
          className="btn btn-success w-full"
        >
          Approve
        </button>
        <button
          onClick={onBlock}
          className="btn btn-danger w-full"
        >
          Block
        </button>
        <button
          onClick={onFork}
          className="btn btn-primary w-full"
        >
          Fork to Human
        </button>
        <button
          onClick={onReplay}
          className="btn btn-secondary w-full"
        >
          Replay Scenario
        </button>
      </div>
    </div>
  )
}
