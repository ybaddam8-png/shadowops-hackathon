import React from 'react'
import { BENCHMARKS, BENCHMARK_CARDS } from '../data/benchmarks'
import { BenchmarkTable } from '../components/BenchmarkTable'

export const Benchmarks: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Benchmark Dashboard</h2>
        <p className="text-slate-400 mt-1">Performance metrics and policy comparison</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {BENCHMARK_CARDS.map((card) => (
          <div key={card.title} className="card">
            <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
              {card.title}
            </p>
            <p className="text-lg font-semibold text-slate-100 mt-2">{card.value}</p>
            <div
              className={`mt-2 px-2 py-1 rounded text-xs font-medium w-fit ${
                card.status === 'verified'
                  ? 'bg-green-900/50 text-green-300'
                  : 'bg-amber-900/50 text-amber-300'
              }`}
            >
              {card.status === 'verified' ? '✓ Verified' : 'Pending'}
            </div>
          </div>
        ))}
      </div>

      <div>
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Policy Comparison</h3>
        <BenchmarkTable rows={BENCHMARKS} />
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
        <p className="text-sm text-slate-300">
          <span className="text-slate-400">📊 Demo Note:</span> GPU model training is pending. Current demo uses verified laptop-safe Q-aware policy.
        </p>
      </div>
    </div>
  )
}
