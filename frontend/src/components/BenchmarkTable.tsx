import React from 'react'
import { BenchmarkRow } from '../data/benchmarks'

export const BenchmarkTable: React.FC<{ rows: BenchmarkRow[] }> = ({ rows }) => {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr className="text-left text-slate-400 text-xs border-b border-slate-700">
            <th className="p-4">Policy</th>
            <th className="p-4">Exact Match</th>
            <th className="p-4">Safety Accuracy</th>
            <th className="p-4">Unsafe Decision Rate</th>
            <th className="p-4">Reward Mean</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.policy} className={`border-b border-slate-800 ${r.isHighlight ? 'bg-slate-800/40' : ''}`}>
              <td className="p-4 font-medium">{r.policy}</td>
              <td className="p-4">{r.exactMatch}%</td>
              <td className="p-4">{r.safetyAccuracy}%</td>
              <td className="p-4">{r.unsafeDecisionRate}%</td>
              <td className="p-4">{r.rewardMean.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
