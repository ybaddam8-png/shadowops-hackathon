import React from 'react'
import { QuarantineTimeline } from '../components/QuarantineTimeline'

export const Quarantine: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Quarantine Resolution</h2>
        <p className="text-slate-400 mt-1">Track and resolve held actions safely</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="col-span-2 lg:col-span-1">
          <QuarantineTimeline />
        </div>

        <div className="card">
          <h3 className="font-semibold text-slate-100 mb-4">Active Quarantines</h3>
          <div className="space-y-4">
            {['GitHub poisoned PR', 'Rogue IAM admin creation', 'Suspicious firewall opening'].map(
              (item, i) => (
                <div key={i} className="p-3 bg-slate-700/50 rounded border border-amber-800">
                  <p className="text-sm text-slate-200">{item}</p>
                  <p className="text-xs text-amber-400 mt-1">Awaiting evidence...</p>
                </div>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
