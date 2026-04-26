import React, { useEffect, useState } from 'react'
import { Activity, AlertCircle, Zap } from 'lucide-react'
import { StatusIndicator } from './DecisionBadge'
import { BackendStatusResponse, getBackendStatus } from '../lib/api'

const DEFAULT_STATUS: BackendStatusResponse = {
  status: 'offline',
  live: false,
  ready: false,
  mode: 'demo_fallback',
  modelLoaded: false,
  classifierLoaded: false,
  adapterLoaded: false,
  startupError: 'Backend status unknown',
  timestamp: new Date().toISOString(),
  source: 'fallback',
}

export const TopBar: React.FC = () => {
  const [backend, setBackend] = useState<BackendStatusResponse>(DEFAULT_STATUS)

  useEffect(() => {
    let alive = true
    const refresh = async () => {
      const status = await getBackendStatus()
      if (alive) setBackend(status)
    }
    refresh()
    const intervalId = window.setInterval(refresh, 10000)
    return () => {
      alive = false
      window.clearInterval(intervalId)
    }
  }, [])

  const sourceLabel = backend.source === 'live' ? 'Live Backend' : 'Fallback Demo Mode'
  const statusLabel = backend.status === 'online' && backend.ready
    ? 'Ready'
    : backend.status === 'online'
      ? 'Online (Not Ready)'
      : 'Offline'
  const statusIndicator = backend.status === 'offline'
    ? 'offline'
    : backend.ready
      ? 'online'
      : 'warning'

  return (
    <div className="fixed top-0 left-56 right-0 h-16 bg-slate-800 border-b border-slate-700 flex items-center px-8 gap-8 z-40">
      <div className="flex-1 flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-sm text-slate-300">
            <span className="font-semibold">Data Source:</span> {sourceLabel}
          </span>
        </div>
        <div className="h-6 w-px bg-slate-700" />
        <div className="flex items-center gap-2">
          <StatusIndicator status={statusIndicator} />
          <span className="text-sm text-slate-300">
            <span className="font-semibold">System Status:</span> {statusLabel}
          </span>
        </div>
        <div className="h-6 w-px bg-slate-700" />
        <div className="text-xs text-slate-400">
          <span className="font-semibold text-slate-300">Mode:</span> {backend.mode}
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-700 rounded-lg">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span className={`text-xs font-mono ${backend.ready ? 'text-green-300' : 'text-amber-300'}`}>
            {backend.ready ? 'READY' : 'FALLBACK'}
          </span>
          <span className="text-xs text-slate-400">Inference</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-700 rounded-lg">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono text-cyan-300">{new Date(backend.timestamp).toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  )
}
