import React from 'react'
import { Scenario } from '../data/scenarios'

export const IncidentReportCard: React.FC<{ scenario: Scenario;
  onCopy?: () => void;
  onExport?: () => void;
  onRun?: () => void;
}> = ({ scenario, onCopy, onExport, onRun }) => {


  return (
    <div className="card">
      <div className="px-6 py-4 border-b border-slate-700">
        <h3 className="font-semibold text-slate-100">Incident Report</h3>
      </div>
      <div className="p-6 space-y-4">
        <div>
          <h4 className="text-sm font-semibold text-slate-200">Summary</h4>
          <p className="text-slate-300 mt-2">{scenario.intent}</p>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-slate-200">Risk Evidence</h4>
          <ul className="list-disc ml-5 mt-2 text-slate-300">
            {scenario.riskVectors.map((v, i) => (
              <li key={i}>{v}</li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-slate-200">Supervisor Decision</h4>
          <p className="text-slate-300 mt-2">{scenario.decision} → FORK TO HUMAN REVIEW</p>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-slate-200">Recommended Action</h4>
          <p className="text-slate-300 mt-2">Require maintainer verification, inspect CI secret permissions, and block merge until package provenance is confirmed.</p>
        </div>

        <div className="flex gap-3 pt-4">
          <button onClick={onCopy} className="btn btn-secondary">Copy Summary</button>
          <button onClick={onExport} className="btn btn-primary">Export Report</button>
          <button onClick={onRun} className="btn btn-success">Run Another Scenario</button>
        </div>
      </div>
    </div>
  )
}
