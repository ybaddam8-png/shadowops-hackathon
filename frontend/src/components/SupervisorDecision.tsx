import React from 'react'
import { Scenario } from '../data/scenarios'
import { DecisionBadge } from './DecisionBadge'
import { BackendStatusResponse, DecisionResponse } from '../lib/api'

interface SupervisorDecisionProps {
  scenario: Scenario
  liveDecision?: DecisionResponse | null
  backendStatus?: BackendStatusResponse | null
  isLoading?: boolean
  onRunLive?: () => void
  onApprove?: () => void
  onBlock?: () => void
  onFork?: () => void
  onReplay?: () => void
}

export const SupervisorDecision: React.FC<SupervisorDecisionProps> = ({
  scenario,
  liveDecision,
  backendStatus,
  isLoading,
  onRunLive,
  onApprove,
  onBlock,
  onFork,
  onReplay,
}) => {
  const decision = liveDecision?.decision || scenario.decision
  const confidence = liveDecision?.confidence ?? scenario.confidence
  const reason = liveDecision?.reason || scenario.reason
  const isLive = liveDecision?.source === 'live'
  const sourceLabel = isLive ? 'LIVE BACKEND RESULT' : 'FALLBACK DEMO RESULT'
  const sourceClass = isLive
    ? 'bg-green-900/20 border-green-800 text-green-300'
    : 'bg-amber-900/20 border-amber-800 text-amber-300'
  const modelMode = liveDecision?.modelMode || backendStatus?.mode || 'demo_fallback'

  return (
    <div className="card space-y-6">
      <div className={`border rounded-lg px-3 py-2 text-xs font-semibold tracking-wide ${sourceClass}`}>
        {sourceLabel}
      </div>

      <div>
        <h3 className="text-xs uppercase tracking-wide font-semibold text-slate-400 mb-3">
          Supervisor Decision
        </h3>
        <div className="flex items-center gap-3">
          <DecisionBadge decision={decision} confidence={confidence} />
          <button
            onClick={onRunLive}
            disabled={isLoading}
            className="btn btn-secondary"
          >
            {isLoading ? 'Running Live Decision...' : 'Run Live Decision'}
          </button>
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
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-cyan-400 font-mono font-semibold">
              {confidence}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2">
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
              Model Mode
            </p>
            <p className="text-sm text-slate-200 mt-1 break-all">{modelMode}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
              Backend Ready
            </p>
            <p className={`text-sm mt-1 ${liveDecision?.backendReady ? 'text-green-300' : 'text-amber-300'}`}>
              {liveDecision?.backendReady ? 'Yes' : 'No (fallback active)'}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
              Risk Score
            </p>
            <p className="text-sm text-slate-200 mt-1">
              {typeof liveDecision?.riskScore === 'number' ? liveDecision.riskScore.toFixed(3) : 'n/a'}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
              Uncertainty
            </p>
            <p className="text-sm text-slate-200 mt-1">
              {typeof liveDecision?.uncertainty === 'number' ? liveDecision.uncertainty.toFixed(3) : 'n/a'}
            </p>
          </div>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-2">
          Reasoning
        </p>
        <p className="text-slate-300 leading-relaxed text-sm">{reason}</p>
        {liveDecision?.safeOutcome && (
          <p className="text-xs text-slate-400 mt-2">
            <span className="font-semibold text-slate-300">Safe outcome:</span> {liveDecision.safeOutcome}
          </p>
        )}
        {liveDecision?.missingEvidence?.length ? (
          <p className="text-xs text-amber-300 mt-2">
            <span className="font-semibold">Missing evidence:</span> {liveDecision.missingEvidence.join(', ')}
          </p>
        ) : null}
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
