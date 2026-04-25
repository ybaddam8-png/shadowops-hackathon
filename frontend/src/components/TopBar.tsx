import React from 'react'
import { Activity, AlertCircle, Zap } from 'lucide-react'
import { StatusIndicator } from './DecisionBadge'

export const TopBar: React.FC = () => {
  return (
    <div className="fixed top-0 left-56 right-0 h-16 bg-slate-800 border-b border-slate-700 flex items-center px-8 gap-8 z-40">
      {/* Left section - Demo info */}
      <div className="flex-1 flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-sm text-slate-300">
            <span className="font-semibold">Demo Mode:</span> Q-Aware Supervisor
          </span>
        </div>
        <div className="h-6 w-px bg-slate-700" />
        <div className="flex items-center gap-2">
          <StatusIndicator status="online" />
          <span className="text-sm text-slate-300">
            <span className="font-semibold">System Status:</span> Online
          </span>
        </div>
      </div>

      {/* Right section - Status metrics */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-700 rounded-lg">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span className="text-xs font-mono text-red-300">0</span>
          <span className="text-xs text-slate-400">Unsafe</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-700 rounded-lg">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono text-cyan-300">GitHub poisoned PR</span>
        </div>
      </div>
    </div>
  )
}
