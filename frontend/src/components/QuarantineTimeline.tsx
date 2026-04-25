import React from 'react'
import { TimelineEvent } from './DecisionBadge'

export const QuarantineTimeline: React.FC = () => {
  const steps = [
    { title: 'Risk detected', status: 'completed' },
    { title: 'Action held', status: 'completed' },
    { title: 'Evidence requested', status: 'completed' },
    { title: 'Context analyzed', status: 'current' },
    { title: 'Final decision issued', status: 'pending' },
  ] as const

  return (
    <div className="card">
      <div className="px-6 py-4 border-b border-slate-700">
        <h3 className="font-semibold text-slate-100">Quarantine Resolution Timeline</h3>
      </div>
      <div className="p-6 space-y-6">
        <p className="text-slate-300">ShadowOps does not guess when confidence is low. It safely pauses the action until enough evidence is available.</p>
        <div className="space-y-6">
          {steps.map((s, idx) => (
            <TimelineEvent
              key={s.title}
              step={idx + 1}
              title={s.title}
              status={s.status as any}
              timestamp={idx === 3 ? new Date().toLocaleString() : undefined}
            />
          ))}
        </div>

        <div className="mt-4 bg-amber-900/10 border border-amber-800 p-4 rounded">
          <h4 className="text-sm font-semibold text-amber-300">Example: Ambiguous pentest request</h4>
          <p className="text-slate-300 mt-2">Initial decision: <span className="font-semibold">QUARANTINE</span></p>
          <p className="text-slate-300">Evidence result: <span className="font-semibold">Authorized test window confirmed</span></p>
          <p className="text-slate-300">Final decision: <span className="font-semibold">ALLOW WITH RESTRICTIONS</span></p>
        </div>
      </div>
    </div>
  )
}
