import React from 'react'
import { MetricCard } from '../components/DecisionBadge'

export const Overview: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl">ShadowOps Mission Control</h2>
          <p className="text-sm text-slate-400 mt-1">AI Cyber Supervisor for Safe Action Decisions</p>
          <p className="text-sm text-slate-400 mt-2 max-w-xl">ShadowOps pauses uncertain actions, gathers evidence, and resolves them safely.</p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Safety Accuracy" value="100%" />
        <MetricCard label="Unsafe Decisions" value={0} />
        <MetricCard label="Exact Match" value="78.1%" />
        <MetricCard label="Active Quarantines" value={3} />
      </div>

      <div className="card">
        <h3 className="font-semibold text-slate-100 mb-3">System Flow</h3>
        <div className="flex items-center gap-6">
          <div className="p-4 bg-slate-800 rounded">Incoming Action</div>
          <div className="text-slate-400">→</div>
          <div className="p-4 bg-slate-800 rounded">Risk Analysis</div>
          <div className="text-slate-400">→</div>
          <div className="p-4 bg-slate-800 rounded">Supervisor Decision</div>
          <div className="text-slate-400">→</div>
          <div className="p-4 bg-slate-800 rounded">Safe Outcome</div>
        </div>
      </div>
    </div>
  )
}
